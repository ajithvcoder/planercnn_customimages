import os
import cv2
import random
import numpy as np
from PIL import Image
from distutils.version import LooseVersion

from sacred import Experiment
from easydict import EasyDict as edict

import torch
import torch.nn.functional as F
import torchvision.transforms as tf

from models.baseline_same import Baseline as UNet
from utils.disp import tensor_to_image
from utils.disp import colors_256 as colors
from bin_mean_shift import Bin_Mean_Shift
from modules import get_coordinate_map
from utils.loss import Q_loss
from instance_parameter_loss import InstanceParameterLoss

ex = Experiment()

folder = './outputs'
index = 0

@ex.main
def predict(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build network
    network = UNet(cfg.model)

    if not (cfg.resume_dir == 'None'):
        model_dict = torch.load(cfg.resume_dir, map_location=lambda storage, loc: storage)
        network.load_state_dict(model_dict)

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    network.to(device)
    network.eval()

    transforms = tf.Compose([
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    bin_mean_shift = Bin_Mean_Shift(device=device)
    k_inv_dot_xy1 = get_coordinate_map(device)
    instance_parameter_loss = InstanceParameterLoss(k_inv_dot_xy1)

    h, w = 192, 256

    focal_length = 517.97
    offset_x = 320
    offset_y = 240

    K = [[focal_length, 0, offset_x],
         [0, focal_length, offset_y],
         [0, 0, 1]]

    K_inv = np.linalg.inv(np.array(K))

    K_inv_dot_xy_1 = np.zeros((3, h, w))

    for y in range(h):
        for x in range(w):
            yy = float(y) / h * 480
            xx = float(x) / w * 640
                
            ray = np.dot(K_inv,
                         np.array([xx, yy, 1]).reshape(3, 1))
            K_inv_dot_xy_1[:, y, x] = ray[:, 0]


    with torch.no_grad():
        image = cv2.imread(cfg.image_path)
        # the network is trained with 192*256 and the intrinsic parameter is set as ScanNet
        image = cv2.resize(image, (w, h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transforms(image)
        image = image.to(device).unsqueeze(0)
        # forward pass
        logit, embedding, _, _, param = network(image)

        prob = torch.sigmoid(logit[0])
        
        # infer per pixel depth using per pixel plane parameter, currently Q_loss need a dummy gt_depth as input
        _, _, per_pixel_depth = Q_loss(param, k_inv_dot_xy1, torch.ones_like(logit))

        # fast mean shift
        segmentation, sampled_segmentation, sample_param = bin_mean_shift.test_forward(
            prob, embedding[0], param, mask_threshold=0.1)

        # since GT plane segmentation is somewhat noise, the boundary of plane in GT is not well aligned, 
        # we thus use avg_pool_2d to smooth the segmentation results
        b = segmentation.t().view(1, -1, h, w)
        pooling_b = torch.nn.functional.avg_pool2d(b, (7, 7), stride=1, padding=(3, 3))
        b = pooling_b.view(-1, h*w).t()
        segmentation = b

        # infer instance depth
        instance_loss, instance_depth, instance_abs_disntace, instance_parameter = instance_parameter_loss(
            segmentation, sampled_segmentation, sample_param, torch.ones_like(logit), torch.ones_like(logit), False)

        # return cluster results
        segmentation = segmentation.cpu().numpy().argmax(axis=1)

        # mask out non planar region
        segmentation[prob.cpu().numpy().reshape(-1) <= 0.1] = 20
        segmentation = segmentation.reshape(h, w)

        # visualization and evaluation
        image = tensor_to_image(image.cpu()[0])
        mask = (prob > 0.1).float().cpu().numpy().reshape(h, w)
        depth = instance_depth.cpu().numpy()[0, 0].reshape(h, w)
        per_pixel_depth = per_pixel_depth.cpu().numpy()[0, 0].reshape(h, w)

        # use per pixel depth for non planar region
        depth = depth * (segmentation != 20) + per_pixel_depth * (segmentation == 20)

        # change non planar to zero, so non planar region use the black color
        segmentation += 1
        segmentation[segmentation == 21] = 0

        pred_seg = cv2.resize(np.stack([colors[segmentation, 0],
                                        colors[segmentation, 1],
                                        colors[segmentation, 2]], axis=2), (w, h))

        # blend image
        blend_pred = (pred_seg * 0.4 + image * 0.6).astype(np.uint8)

        mask = cv2.resize((mask * 255).astype(np.uint8), (w, h))
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # visualize depth map as PlaneNet
        depth = 255 - np.clip(depth / 5 * 255, 0, 255).astype(np.uint8)
        depth = cv2.cvtColor(cv2.resize(depth, (w, h)), cv2.COLOR_GRAY2BGR)

        image_c = np.concatenate((image, pred_seg, blend_pred, mask, depth), axis=1)

    imageFilename = str(index) + '_model_texture.png'
    cv2.imwrite(folder + '/' + imageFilename, image_c)

    # create face from segmentation
    faces = []
    for y in range(h-1):
        for x in range(w-1):
            segmentIndex = segmentation[y, x]
            # ignore non planar region
            if segmentIndex == 0:
                continue

            # add face if three pixel has same segmentatioin
            depths = [depth[y][x], depth[y + 1][x], depth[y + 1][x + 1]]
            if segmentation[y + 1, x] == segmentIndex and segmentation[y + 1, x + 1] == segmentIndex and np.array(depths).min() > 0 and np.array(depths).max() < 10:
                faces.append((x, y, x, y + 1, x + 1, y + 1))

            depths = [depth[y][x], depth[y][x + 1], depth[y + 1][x + 1]]
            if segmentation[y][x + 1] == segmentIndex and segmentation[y + 1][x + 1] == segmentIndex and np.array(depths).min() > 0 and np.array(depths).max() < 10:
                faces.append((x, y, x + 1, y + 1, x + 1, y))


    with open(folder + '/' + str(index) + '_model.ply', 'w') as f:
        header = """ply
format ascii 1.0
comment VCGLIB generated
comment TextureFile """
        header += imageFilename
        header += """
element vertex """
        header += str(h * w)
        header += """
property float x
property float y
property float z
property uchar red                                     { start of vertex color }
property uchar green
property uchar blue
element face """
        header += str(len(faces))
        header += """
property list uchar int vertex_indices
property list uchar float texcoord
end_header
"""
        f.write(header)
        for y in range(h):
            for x in range(w):
                segmentIndex = segmentation[y][x]
                if segmentIndex == 20:
                    f.write("0.0 0.0 0.0\n")
                    continue
                ray = K_inv_dot_xy_1[:, y, x]
                X, Y, Z = ray * depth[y, x]
                R, G, B = image[y,x]
                f.write(str(X) + ' ' + str(Y) + ' ' + str(Z) + ' ' + str(R) + ' ' + str(G) + ' ' + str(B) + '\n')

        for face in faces:
            f.write('3 ')
            for c in range(3):
                f.write(str(face[c * 2 + 1] * w + face[c * 2]) + ' ')
            f.write('6 ')
            for c in range(3):
                f.write(str(float(face[c * 2]) / w) + ' ' + str(1 - float(face[c * 2 + 1]) / h) + ' ')
            f.write('\n')
        f.close()
        pass
    return

if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    ex.add_config('./configs/predict.yaml')
    ex.run_commandline()