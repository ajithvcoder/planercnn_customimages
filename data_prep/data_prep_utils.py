"""
Copyright (c) 2017 Matterport, Inc.
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import os
import math
import random
import numpy as np
import torch
import cv2
import itertools


## Segmentation - color codes used similar to scene_00000/annotation/segmentation images which we use to get from parse.py
class SegColorPalette:
    def __init__(self, numColors):
        self.colorMap = np.array([[160, 15, 0],
                                  [248, 17, 0],
                                  [140, 10, 0],
                                  [48, 17, 0], [236, 19, 0], [60, 15, 0], [20, 30, 0], [172, 13, 0], 
                                  [16, 14, 0], [128, 12, 0], [4, 16, 0], [8, 7, 0], [216, 14, 0], 
                                  [204, 16, 0], 
                                  [252, 33, 0], [28, 12, 0], [4, 5, 6], [104, 16, 0], 
                                  [228, 12, 0], [72, 13, 0], [148, 17, 0], [80, 20, 0], [52, 8, 0]
                                  
        ], dtype=np.uint8)

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate([self.colorMap, np.random.randint(255, size = (numColors - self.colorMap.shape[0], 3), dtype=np.uint8)], axis=0)
            pass

        return

    def getColorMap(self, returnTuples=False):
        if returnTuples:
            return [tuple(color) for color in self.colorMap.tolist()]
        else:
            return self.colorMap

    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3), dtype=np.uint8)
        else:
            return self.colorMap[index]
        pass    

## Draw segmentation image. The input could be either HxW or HxWxC
def drawSegmentationImage(segmentations, numColors=42, blackIndex=-1, blackThreshold=-1):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2)
    else:
        if blackThreshold > 0:
            segmentations = np.concatenate([segmentations, np.ones((segmentations.shape[0], segmentations.shape[1], 1)) * blackThreshold], axis=2)
            blackIndex = segmentations.shape[2] - 1
            pass

        numColors = max(numColors, segmentations.shape[2] + 2)
        pass
    randomColor = SegColorPalette(numColors).getColorMap()
    if blackIndex >= 0:
        randomColor[blackIndex] = 0
        pass
    width = segmentations.shape[1]
    height = segmentations.shape[0]
    if segmentations.ndim == 3:
        segmentation = np.argmax(segmentations, 2)
    else:
        segmentation = segmentations
        pass

    segmentation = segmentation.astype(np.int32)
    return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))


# gives color codes of r,g,b used in a image
def extractColorCodes(image,print_codes=0):
    print(image.shape)
    imCodes=set()
    if len(image.shape)==2:
        w,h = image.shape
        for i in range(w):
            for j in range(h):
                a= image[i][j]
                imCodes.add((a))
        #print(image[i])
            

    if len(image.shape) ==3:
        w,h,_ = image.shape
    
    
        for i in range(w):
            for j in range(h):
                a,b,c = image[i][j]
                imCodes.add((a,b,c))
        #print(len(imCodes))
        if print_codes==1:
            print(imCodes)
    return len(imCodes),imCodes


## Depth - color codes used similar to scene_00000/frames/depth images which we use to get from original scannet dataset
class DepthColorPalette:
    def __init__(self, numColors):
        self.colorMap = np.array([ [3, 3, 3], [6, 6, 6], [5, 5, 5], [4, 4, 4], [7, 7, 7], [8, 8, 8],[0,0,0]
                                  
        ], dtype=np.uint8)

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate([self.colorMap, np.random.randint(255, size = (numColors - self.colorMap.shape[0], 3), dtype=np.uint8)], axis=0)
            pass

        return

    def getColorMap(self, returnTuples=False):
        if returnTuples:
            return [tuple(color) for color in self.colorMap.tolist()]
        else:
            return self.colorMap

    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3), dtype=np.uint8)
        else:
            return self.colorMap[index]
        pass    

## Draw segmentation image. The input could be either HxW or HxWxC
def drawDepthImage(segmentations, numColors=42, blackIndex=-1, blackThreshold=-1):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2)
    else:
        if blackThreshold > 0:
            segmentations = np.concatenate([segmentations, np.ones((segmentations.shape[0], segmentations.shape[1], 1)) * blackThreshold], axis=2)
            blackIndex = segmentations.shape[2] - 1
            pass

        numColors = max(numColors, segmentations.shape[2] + 2)
        pass
    randomColor = DepthColorPalette(numColors).getColorMap()
    if blackIndex >= 0:
        randomColor[blackIndex] = 0
        pass
    width = segmentations.shape[1]
    height = segmentations.shape[0]
    if segmentations.ndim == 3:
        segmentation = np.argmax(segmentations, 2)
    else:
        segmentation = segmentations
        pass

    segmentation = segmentation.astype(np.int32)
    return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))









