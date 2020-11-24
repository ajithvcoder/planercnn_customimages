import cv2 
import numpy as np 
depthPath = "Input_images/1_depth_scannet.png"
#depthPath1 = "Input_images/9_midas_modified2.png"
depthPath1 = "Input_images/1_midas.png"

im= cv2.imread(depthPath)
#print(im)
# cv2.imshow("im",im)
# cv2.waitKey(0)
# cv2.destroyWindows(0)
print(im.shape)

depth = cv2.imread(depthPath, -1).astype(np.float32) / 1000.00
print(depth)
print(depth.shape)

depth1 = cv2.imread(depthPath1, -1).astype(np.float32) / 40000.00
print(depth1)
print(depth1.shape)