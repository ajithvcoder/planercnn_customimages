"""
In scannet depth color code used are : {(3, 3, 3), (6, 6, 6), (5, 5, 5), (4, 4, 4), (0, 0, 0), (7, 7, 7), (8, 8, 8)}
Below codes can be modified as functionals but 1000s of images are ivolved hence i left like below commenting method for better understanding
"""

# # Section 1: used for finding total or all color codes used in midas (any images)
import cv2
from data_prep_utils import extractColorCodes

image = cv2.imread("Input_images/pil_output.png",-1)
# image = cv2.imread("Input_images/1_depth_scannet.png")
# print(image.shape)
# image = cv2.imread("Input_images/1_depth_scannet.png",-1)
if image.any():
    num, codes = extractColorCodes(image)

    print(num)
    print(codes)
else:
    print("no image")

# 256 different color codes are used in Midas outputs so 
# in multiples of 7 convert all range of colors within {(3, 3, 3), (6, 6, 6), (5, 5, 5), (4, 4, 4), (0, 0, 0), (7, 7, 7), (8, 8, 8)}




# ########## Section 2:  Single image depth transformation

# import cv2 
# import numpy as np
# # #imagePath = "plane_masks/0_plane_masks_0.npy"
# imagePath = "Input_images/1_midas.png"

# image = cv2.imread(imagePath)
# imageAlpha = cv2.imread(imagePath,-1)

# w,h,_ = image.shape 

# b,g,r = cv2.split(image)
# b = b//35
# g = g//35
# r = r//35
# print(b.shape)
# print(imageAlpha.shape)
# #alpha_channel = np.ones(b.shape, dtype=b.dtype) * 50
# modImage = cv2.merge((b,g,r))
# rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
# rgba[:, :, 3] = imageAlpha

# # for i in range(w):
# #     for j in range(h):
# #         b,g,r = image[i][j]


# #         value = int((b+600)//35)       
# #         #This is for normalization with normal human eyes compared with scannet depth and midas net depth images and made it
# #         # feel free to change above formula
# #         image[i][j] = value,value,value,imageAlpha[i]
# print("completed")

# # new_seg.shape
# cv2.imshow("seg",rgba)

# cv2.imwrite("Input_images/segment_output_1.png",rgba)
# cv2.waitKey(0)
# cv2.destroyWindow(0)


# from PIL import Image
# import numpy as np

# FNAME = "Input_images/1_midas.png"
# img = Image.open(FNAME).convert('RGBA')
# x = np.array(img)
# r, g, b, a = np.rollaxis(x, axis = -1)
# r = r//35
# g = g//35
# b = b//35
# #print(r.shape)
# x = np.dstack([r, g, b, a])
# img = Image.fromarray(x, 'RGBA')

# img.show()
# img.save('Input_images/pil_output.png')



########## Section 3:  Multi image depth transformation

# import cv2 
# import numpy as np
# import os 
# # #imagePath = "plane_masks/0_plane_masks_0.npy"
# dest_segmentation = "output_images/depth_midas_modified2/"

# src_segmentation = "Input_images/midas/"


# if __name__ == '__main__':
#     dirs = os.listdir( src_segmentation )
#     #print(dirs)
#     count = 0
#     for item in dirs:
#         if os.path.isfile(src_segmentation+item):
#             if ".png" in item:
#                 image = cv2.imread(src_segmentation+item)
#                 w,h,_ = image.shape 
#                 for i in range(w):
#                     for j in range(h):
#                         b,g,r = image[i][j]
#                         #value = int((b+600)//35)  
#                         value = int((b)//35)         
#                         #This is for normalization: with normal human eyes compared with scannet depth and midas net depth images and made it
#                         # feel free to change above formula
#                         image[i][j] = value,value,value
#                 cv2.imwrite(dest_segmentation+item,image)

                
