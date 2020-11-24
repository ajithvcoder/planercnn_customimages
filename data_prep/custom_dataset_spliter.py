"""
purpose: From a directory where all segmentations,depth,color are present this script helps to seperate everything
Please execute eeach section one by one to avoid mass deletion/alterations
"""

import shutil
import os 

####################### Section : 1 ################################################

## Copying segmentations to annotations/segmentations

# dest_segmentation = "frames_cus_scene0003_02/annotations/segmentations/" 

# src_segmentation = "test/inference/"

# if __name__ == '__main__':
#     dirs = os.listdir( src_segmentation )
#     #print(dirs)
#     for item in dirs:
#         if os.path.isfile(src_segmentation+item):
#             if "segmentation_0_final.png" in item:
#                 #print(src_segmentation + item, dest_segmentation+item)
#                 shutil.copy(src_segmentation + item, dest_segmentation+item)

########Section : 2 #######################################################################


## Copying frame/color images 

# dest_segmentation = "frames_cus_scene0003_02/color/" 

# src_segmentation = "test/inference/"

# if __name__ == '__main__':
#     dirs = os.listdir( src_segmentation )
#     #print(dirs)
#     for item in dirs:
#         if os.path.isfile(src_segmentation+item):
#             if "image_0.png" in item:
#                 #print(src_segmentation + item, dest_segmentation+item)
#                 shutil.copy(src_segmentation + item, dest_segmentation+item)

#####Section : 3###################################################################################

## copying frame/depth images


# dest_segmentation = "frames_cus_scene0003_02/depth/" 

# src_segmentation = "test/inference/"

# if __name__ == '__main__':
#     dirs = os.listdir( src_segmentation )
#     #print(dirs)
#     for item in dirs:
#         if os.path.isfile(src_segmentation+item):
#             if "depth_0_final.png" in item:
#                 #print(src_segmentation + item, dest_segmentation+item)
#                 shutil.copy(src_segmentation + item, dest_segmentation+item)


#####Section : 4###################################################################################

## copying plane_masks


dest_segmentation = "plane_masks_score_large/" 

src_segmentation = "test_large_save/inference/"

if __name__ == '__main__':
    dirs = os.listdir( src_segmentation )
    #print(dirs)
    count = 0
    for item in dirs:
        if os.path.isfile(src_segmentation+item):
            if "plane_masks_0.npy" in item:
                count += 1
                if count>1344:
                    exit()
                #print(src_segmentation + item, dest_segmentation+item)
                shutil.copy(src_segmentation + item, dest_segmentation+item)


