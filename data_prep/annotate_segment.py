from data_prep_utils import drawSegmentationImage
import cv2 
import numpy as np
imagePath = "plane_masks/0_plane_masks_0.npy"

planes_mask = np.load(imagePath)
masks = np.concatenate([np.maximum(1 - planes_mask.sum(0, keepdims=True), 0), planes_mask], axis=0).transpose((1, 2, 0))
new_seg = drawSegmentationImage(masks,blackIndex=0)
new_seg.shape
cv2.imshow("seg",new_seg)

#cv2.imwrite("segment_output.png",new_seg)
cv2.waitKey(0)
cv2.destroyWindow(0)

