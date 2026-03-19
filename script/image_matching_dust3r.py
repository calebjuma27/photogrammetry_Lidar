#-------------------------------------------------------------------------------
# Name:        image_matching_dust3r
# Purpose:      use dust3r to create a depth map
#
# Author:      caleb +ai
#
# Created:     19/03/2026
# Copyright:   (c) caleb 2026
# Licence:     <your licence>
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
import cv2
import numpy as np

# 1. Load your images

image_1=r"C:\Users\caleb\OneDrive\Desktop\other\codebase_2026\photogrammetry_Lidar\dense_image_matching\input_images\20260319_081743.jpg"
image_2=r"C:\Users\caleb\OneDrive\Desktop\other\codebase_2026\photogrammetry_Lidar\dense_image_matching\input_images\20260319_081746.jpg"
img1 = cv2.imread(image_1, 0) # Load as grayscale
img2 = cv2.imread(image_2, 0)


## to do. Dust3r helps to orient and generate a 3D image without knowledge of intrinisc or extrinsic parameters. It is a vison transformer
