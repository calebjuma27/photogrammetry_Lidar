#-------------------------------------------------------------------------------
# Name:        Dense image matching
# Purpose:
#
# Author:      caleb +AI
#
# Created:     19/03/2026
# Copyright:   (c) caleb 2026
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
import cv2
import numpy as np

#Data

image_1=r"C:\Users\caleb\OneDrive\Desktop\other\codebase_2026\photogrammetry_Lidar\dense_image_matching\input_images\20260319_081746.jpg"
image_2=r"C:\Users\caleb\OneDrive\Desktop\other\codebase_2026\photogrammetry_Lidar\dense_image_matching\input_images\20260319_081748.jpg"

ouptut_3D_points=r"C:\Users\caleb\OneDrive\Desktop\other\codebase_2026\photogrammetry_Lidar\dense_image_matching\output_3D_points\output.ply"


# 1. Load your images
img1 = cv2.imread(image_1, 0) # Load as grayscale
img2 = cv2.imread(image_2, 0)

#cv2.imshow('Display WIndow',img1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# 2. Find keypoints and descriptors (ORB)
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

img_with_keys=cv2.drawKeypoints(img1,kp1,None,color=(0,255,0))
#cv2.imshow('Display WIndow',img_with_keys)
#cv2.waitKey(0)

# 3. Match features using Brute-Force
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

# 4. Extract location of good matches
pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

# 5. Find Fundamental Matrix and Rectify
# This mathematically aligns the images so search is horizontal
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
_, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, img1.shape[::-1])

'''
Success Flag (a Boolean: True or False indicating if it worked).
H1 (The rectification matrix for the first image).
H2 (The rectification matrix for the second image).
In Python, the underscore (_) is used as a throwaway variable.
It tells the interpreter: "This function returns a value here, but I don't plan on using it, so don't bother giving it a name."
# By using _, you are effectively saying: "I only care about the two matrices (H1 and H2);
I'm going to ignore whether the function technically reported a 'success' or 'failure' status.
'''

#applies the transformations (H1 and H2) to the actual image pixels.
img1_rect = cv2.warpPerspective(img1, H1, img1.shape[::-1])
img2_rect = cv2.warpPerspective(img2, H2, img2.shape[::-1])


# 6. Compute Dense Matching (Disparity Map)
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=64,  # Must be divisible by 16
    blockSize=5,
    P1=8 * 3 * 5**2,
    P2=32 * 3 * 5**2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disparity = stereo.compute(img1_rect, img2_rect)



# Normalize for visualization
disparity_vis = cv2.normalize(disparity, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

##cv2.imshow('Dense Matching (Disparity)', disparity_vis)
##cv2.waitKey(0)



# 2. Apply a ColorMap (JET is the classic "Thermal" look)
# Red = Very Close, Blue = Very Far
disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)

# 3. Show the result
##cv2.imshow('Depth Map (Heatmap)', disparity_color)
##cv2.waitKey(0)
##cv2.destroyAllWindows()



### HAs error

# convert disparity map to a 3D point cloud

#create a dummy reprojection matrix Q
h, w = img1_rect.shape[:2]
# Approximate Q: [1, 0, 0, -w/2], [0, 1, 0, -h/2], [0, 0, 0, focal_length], [0, 0, 1/baseline, 0]

# Using a generic focal length (usually ~0.8 * width)
f = 9583 #generic==>#0.8 * w
print("h,w is", h,w)
print("focal length is", f)
baseline=0.03

Q = np.float32([[1, 0, 0, -0.5*w],
                [0, -1, 0,  -0.5*h], # Flip Y so 3D coordinates aren't upside down
                [0, 0, 0,  f],
                [0, 0, -1/baseline,  0]])

#Project to 3D and Save
#We use cv2.reprojectImageTo3D to generate the coordinates and then filter out the "noise" (points where disparity was too low or invalid).

# 1. Generate the 3D points
points_3d = cv2.reprojectImageTo3D(disparity, Q)

# 2. Get the colors from the original rectified image
colors = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2RGB)

# 3. Filter out points with invalid disparity (usually <= 0 or very small)
#mask = disparity > disparity.min()
mask = (disparity > disparity.min()) & np.isfinite(points_3d).all(axis=2)
out_points = points_3d[mask]
out_colors = colors[mask]

#OR


# 3. Create the corrected PLY file
def save_ply_fixed(filename, points, colors):
    # Flatten and combine points (X, Y, Z) and colors (R, G, B)
    # Ensure they are the exact same length
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([points, colors])

    # Calculate the EXACT number of vertices to put in the header
    num_verts = len(verts)

    header = (
        f"ply\n"
        f"format ascii 1.0\n"
        f"element vertex {num_verts}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"end_header\n"
    )

    with open(filename, 'w') as f:
        f.write(header)
        # Use space delimiter and specific formatting for X, Y, Z (float) and R, G, B (int)
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')


# 4. Define a helper function to save as .PLY (readable by MeshLab/CloudCompare)
def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\n"
                 f"property float x\nproperty float y\nproperty float z\n"
                 f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 f"end_header\n").encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d')







save_ply_fixed(ouptut_3D_points, out_points, out_colors)
print(f"Saved {len(out_points)} vertices to cleaned_cloud.ply")

##write_ply(ouptut_3D_points, out_points, out_colors)
##print("Point cloud saved as out.ply")

