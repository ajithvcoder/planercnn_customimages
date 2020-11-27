"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import numpy as np
import glob
import cv2
import os

from utils import *

class CustomScene():
    """ This class handle one scene of the scannet dataset and provide interface for dataloaders """
    def __init__(self, options, scenePath, scene_id, confident_labels, layout_labels, load_semantics, load_boundary=False):
        self.options = options
        self.load_semantics = load_semantics
        self.load_boundary = load_boundary
        #self.scannetVersion = 2

        self.confident_labels, self.layout_labels = confident_labels, layout_labels
        
        self.camera = np.zeros(6)

        # reading focal length of the camera, with heigh and depth width and height, num of depth frames
        with open(scenePath + '/' + scene_id + '.txt') as f:
            for line in f:
                line = line.strip()
                tokens = [token for token in line.split(' ') if token.strip() != '']
                if tokens[0] == "fx_depth":
                    self.camera[0] = float(tokens[2])
                if tokens[0] == "fy_depth":
                    self.camera[1] = float(tokens[2])
                if tokens[0] == "mx_depth":
                    self.camera[2] = float(tokens[2])                            
                if tokens[0] == "my_depth":
                    self.camera[3] = float(tokens[2])                            
                elif tokens[0] == "colorWidth":
                    self.colorWidth = int(tokens[2])
                elif tokens[0] == "colorHeight":
                    self.colorHeight = int(tokens[2])
                elif tokens[0] == "depthWidth":
                    self.depthWidth = int(tokens[2])
                elif tokens[0] == "depthHeight":
                    self.depthHeight = int(tokens[2])
                elif tokens[0] == "numDepthFrames":
                    self.numImages = int(tokens[2])
                    pass
                continue
            pass
        self.depthShift = 40000.0
        self.imagePaths = [scenePath + '/frames/color/' + str(imageIndex) + '.jpg' for imageIndex in range(self.numImages - 1)]
            #pass
            
        self.camera[4] = self.depthWidth
        self.camera[5] = self.depthHeight
        self.planes = np.load(scenePath + '/annotation/planes.npy')
        print("planes",self.planes)

        #self.plane_info = np.load(scenePath + '/annotation/plane_info.npy')            
        # if len(self.plane_info) != len(self.planes):
        #     print('invalid number of plane info', scenePath + '/annotation/planes.npy', scenePath + '/annotation/plane_info.npy', len(self.plane_info), len(self.planes))
        #     exit(1)

        self.scenePath = scenePath
        return
        

    def transformPlanes(self, transformation, planes):
        planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
        
        centers = planes
        centers = np.concatenate([centers, np.ones((planes.shape[0], 1))], axis=-1)
        newCenters = np.transpose(np.matmul(transformation, np.transpose(centers)))
        newCenters = newCenters[:, :3] / newCenters[:, 3:4]

        refPoints = planes - planes / np.maximum(planeOffsets, 1e-4)
        refPoints = np.concatenate([refPoints, np.ones((planes.shape[0], 1))], axis=-1)
        newRefPoints = np.transpose(np.matmul(transformation, np.transpose(refPoints)))
        newRefPoints = newRefPoints[:, :3] / newRefPoints[:, 3:4]

        planeNormals = newRefPoints - newCenters
        planeNormals /= np.linalg.norm(planeNormals, axis=-1, keepdims=True)
        planeOffsets = np.sum(newCenters * planeNormals, axis=-1, keepdims=True)
        newPlanes = planeNormals * planeOffsets
        return newPlanes

    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, imageIndex):
        imagePath = self.imagePaths[imageIndex]
        image = cv2.imread(imagePath)

        # segmentation is read
        segmentationPath = imagePath.replace('frames/color/', 'annotation/segmentation/').replace('.jpg', '.png')

        # depth is read
        depthPath = imagePath.replace('color', 'depth').replace('.jpg', '.png')
        # pose is read
        posePath = imagePath.replace('color', 'pose').replace('.jpg', '.txt')
        # semanticspath is read
        semanticsPath = imagePath.replace('color/', 'instance-filt/').replace('.jpg', '.png')            


        try:
            #print("depthshit value",self.depthShift)
            self.depthShift = 40000.00 
            # Alpha value is read
            depth = cv2.imread(depthPath, -1).astype(np.float32) / self.depthShift
            print("depthvalue",depth)
            #np.save("depthdepth)
        except:
            print('no depth image', depthPath, self.scenePath)
            exit(1)

        # extrinsic values are got from pose path
        extrinsics_inv = []
        with open(posePath, 'r') as f:
            for line in f:
                extrinsics_inv += [float(value) for value in line.strip().split(' ') if value.strip() != '']
                continue
            pass
        extrinsics_inv = np.array(extrinsics_inv).reshape((4, 4))
        extrinsics = np.linalg.inv(extrinsics_inv)

        temp = extrinsics[1].copy()
        extrinsics[1] = extrinsics[2]
        extrinsics[2] = -temp

        # segmentation is read with alpha values
        segmentation = cv2.imread(segmentationPath, -1).astype(np.int32)
        

        segmentation = (segmentation[:, :, 2] * 256 * 256 + segmentation[:, :, 1] * 256 + segmentation[:, :, 0]) // 100 - 1

        segments, counts = np.unique(segmentation, return_counts=True)
        segmentList = zip(segments.tolist(), counts.tolist())
        segmentList = [segment for segment in segmentList if segment[0] not in [-1, 167771]]
        segmentList = sorted(segmentList, key=lambda x:-x[1])
        print("0002")
        newPlanes = []
        newPlaneInfo = []
        newSegmentation = np.full(segmentation.shape, fill_value=-1, dtype=np.int32)
        # print("segment list shape",segmentList)
        # if True:
        #     exit()

        newIndex = 0
        for oriIndex, count in segmentList:
            if count < self.options.planeAreaThreshold:
                continue
            if oriIndex >= len(self.planes):
                continue            
            if np.linalg.norm(self.planes[oriIndex]) < 1e-4:
                continue
            # print("stuck here ",count," ",oriIndex)
            # print(self.planes[oriIndex])
            newPlanes.append(self.planes[oriIndex])
            #print(len(newPlanes))
            newSegmentation[segmentation == oriIndex] = newIndex
            #newPlaneInfo.append(self.plane_info[oriIndex] + [oriIndex])
            newIndex += 1
            continue

        segmentation = newSegmentation
        planes = np.array(newPlanes)
        #plane_info = newPlaneInfo        

        image = cv2.resize(image, (depth.shape[1], depth.shape[0]))
        # print("length of 1st plane",len(planes))
        # print("reached -ckpt1")
        if len(planes) > 0:
            planes = self.transformPlanes(extrinsics, planes)
            print("reached -ckpt10")
            #print("depth----",depth)
            segmentation, plane_depths = cleanSegmentation(image, planes, segmentation, depth, self.camera, planeAreaThreshold=self.options.planeAreaThreshold, planeWidthThreshold=self.options.planeWidthThreshold, confident_labels=self.confident_labels, return_plane_depths=True)
            print("reached -ckpt12")
            masks = (np.expand_dims(segmentation, -1) == np.arange(len(planes))).astype(np.float32)
            plane_depth = (plane_depths.transpose((1, 2, 0)) * masks).sum(2)
            plane_mask = masks.max(2)
            plane_mask *= (depth > 1e-4).astype(np.float32)            
            plane_area = plane_mask.sum()
            print("reached -ckpt13")
            depth_error = (np.abs(plane_depth - depth) * plane_mask).sum() / max(plane_area, 1)
            # if depth_error > 0.1:
            #     print('depth error', depth_error)
            #     planes = []
            #     pass

            pass
        
        if len(planes) == 0 or segmentation.max() < 0:
            print("plane length", len(planes))
            print("seg",segmentation.max() < 0)
            print("reached -ckpt2")
            exit(1)
            pass
        print("reached -ckpt21555")
        info = [image, planes, segmentation, depth, self.camera, extrinsics]
        
        if self.load_semantics or self.load_boundary:
            semantics = cv2.imread(semanticsPath, -1).astype(np.int32)
            semantics = cv2.resize(semantics, (640, 480), interpolation=cv2.INTER_NEAREST)
            info.append(semantics)
        else:
            info.append(0)
            pass
        print("check #----------------------------------------------------------")
        if self.load_boundary:
            print("reached -ckpt3")
            plane_points = []
            plane_instances = []
            for plane_index in range(len(planes)):            
                ys, xs = (segmentation == plane_index).nonzero()
                if len(ys) == 0:
                    plane_points.append(np.zeros(3))
                    plane_instances.append(-1)
                    continue
                u, v = int(round(xs.mean())), int(round(ys.mean()))
                depth_value = plane_depths[plane_index, v, u]
                point = np.array([(u - self.camera[2]) / self.camera[0] * depth_value, depth_value, -(v - self.camera[3]) / self.camera[1] * depth_value])
                plane_points.append(point)
                plane_instances.append(np.bincount(semantics[ys, xs]).argmax())
                continue

            # for plane_index in range(len(planes)):
            #     if plane_info[plane_index][0][1] in self.layout_labels:
            #         semantics[semantics == plane_instances[plane_index]] = 65535
            #         plane_instances[plane_index] = 65535
            #         pass
            #     continue
            
            parallelThreshold = np.cos(np.deg2rad(30))
            boundary_map = np.zeros(segmentation.shape)
            
            plane_boundary_masks = []
            print("reached -ckpt5")
            for plane_index in range(len(planes)):
                mask = (segmentation == plane_index).astype(np.uint8)
                plane_boundary_masks.append(cv2.dilate(mask, np.ones((3, 3)), iterations=15) - cv2.erode(mask, np.ones((3, 3)), iterations=15) > 0.5)
                continue
                        

            for plane_index_1 in range(len(planes)):
                plane_1 = planes[plane_index_1]
                offset_1 = np.linalg.norm(plane_1)
                normal_1 = plane_1 / max(offset_1, 1e-4)                                
                for plane_index_2 in range(len(planes)):
                    if plane_index_2 <= plane_index_1:
                        continue
                    if plane_instances[plane_index_1] != plane_instances[plane_index_2] or plane_instances[plane_index_1] == -1:
                        continue
                    plane_2 = planes[plane_index_2]                    
                    offset_2 = np.linalg.norm(plane_2)
                    normal_2 = plane_2 / max(offset_2, 1e-4)            
                    if np.abs(np.dot(normal_2, normal_1)) > parallelThreshold:
                        continue
                    point_1, point_2 = plane_points[plane_index_1], plane_points[plane_index_2]
                    if np.dot(normal_1, point_2 - point_1) <= 0 and np.dot(normal_2, point_1 - point_2) < 0:
                        concave = True
                    else:
                        concave = False
                        pass
                    boundary_mask = (plane_depths[plane_index_1] < plane_depths[plane_index_2]).astype(np.uint8)
                    boundary_mask = cv2.dilate(boundary_mask, np.ones((3, 3)), iterations=5) - cv2.erode(boundary_mask, np.ones((3, 3)), iterations=5)
                    instance_mask = semantics == plane_instances[plane_index_1]
                    boundary_mask = np.logical_and(boundary_mask > 0.5, instance_mask)
                    boundary_mask = np.logical_and(boundary_mask, np.logical_and(plane_boundary_masks[plane_index_1], plane_boundary_masks[plane_index_2]))
                    if concave:
                        boundary_map[boundary_mask] = 1
                    else:
                        boundary_map[boundary_mask] = 2
                    continue
                continue
            print("reached -ckpt5")
            info[-1] = boundary_map
            pass
        print(len(info))
        if True:
            exit()
        return info
