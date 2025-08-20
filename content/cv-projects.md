---
title: 'Computer Vision Projects'
description: 'Color segmentation, panorama stitching, two-view 3d reconstruction and depth estimation projects'
image: 'cv-projects/cv-projects.jpg'
preview: 'cv-projects/preview.gif'
priority: 4
tags:
  - Python
  - OpenCV
links:
  - text: View on GitHub
    href: https://github.com/ifuller140/computer_vision_projects
---

## Project 1: Color Segmentation using GMM

```python
# Download training images from Google Drive
import gdown
gdown.download_folder(id="18Mx2Xc9UNFZYajYu9vfmRFlFCcna5I0J", quiet=True, use_cookies=False)

# Download testing images from Google Drive
gdown.download_folder(id="1Yl4_5O_ZEkz_KJVs0_vS5TrZUqMYkwr4", quiet=True, use_cookies=False)

# Check whether the training images were successfully imported
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

train_image = mpimg.imread('/content/train_images/106.jpg')
plt.imshow(train_image)
plt.axis("off")
plt.show()

import cv2
import os
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt


# TODO: Read in training images
path = "/content/train_images"
files = "*.jpg"

file_path = glob.glob(os.path.join(path, files))
print(file_path)
total = []

# TODO: Iterate over training images to extract orange pixels using masks
for x in file_path:
    image = cv2.imread(x) #read image x
    image_array = np.array(image) #convert the image into an np array
    # lower = np.array([10, 100, 20]) #a BGR lowerbound, this is a dark green
    # upper = np.array([25, 255, 255]) #a BGR upperbound, this is a bright yellow

    #mostly just the ball (ONLY WORKS WHEN WE ASSUME IMAGE IS RGB, BUT IT
      # ACTUALLY COMES IN AS BGR)
    lower = np.array([110, 100, 100])
    upper = np.array([120, 255, 255])


    #trying to improve on last one

    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV) #create new copy of image w/ HSV color space
    mask = cv2.inRange(hsv,lower,upper) #grab the subset of the image who's color lies within the given bound
      #Question about previous line: does mask just tell you which pixels fall in the specified range?
        #Answer - The mask will use the lower and upper bounds given by the array of color values and matches pixels based on those ranges.


    # write original image to memory
    cv2.imwrite("x.jpg", image)
    # write the mask to memeory
    cv2.imwrite('mask.jpg', mask)

    #write hsv
    cv2.imwrite('hsv.jpg', cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))

    #write image_masked.jpg
    image_masked = cv2.bitwise_and(image, image, mask=mask)
    cv2.imwrite('image_masked.jpg', image_masked)

    orange_total = hsv[mask > 0]
    total.extend(orange_total)


total = np.array(total)
print(total)
print(f"total.shape: {total.shape}")
print(f"total.size: {total.size}")



# TODO: Compute mean and covariance using MLE(Maximum Likelihood Estimation)
mean = np.mean(total, axis = 0)
print("mean: ")
print(mean)
covariance = np.cov(total.T)
# covariance = np.cov(total)

print(covariance)
# TODO: Compute PDF(Probability Density Function) of single gaussian model

def pdf(x, mean, covariance):
  diff = (x - mean).reshape(3,1)  #(X - u) part of the equatation


  #FOR DEBUGGING
  # left = np.dot(x_m, np.linalg.inv(covariance))
  # print("LEFT: ")
  # print(left)

  # right = x_m.T
  # print("RIGHT: ")
  # print(right)
  #FOR DEBUGGING

  # exp_arg = -0.5 * np.dot(np.dot(x_m.T, np.linalg.inv(covariance)), x_m)
  # print(exp_arg[0][0])
  # exp = np.exp(np.clip(exp_arg[0][0], -700, 700))
  # this represents the exp part of the equation.
  #x_m.T represents the transpose of (x-u)
  # np.linalg.inv(covariance) represents the inverse of the covariance matrix.
  #All multiplied by -1/2

  #determinate is maxed b/c determinate can experience underflow making it default to 0 & breaking the sqrt function
  # pdf = exp / np.sqrt((2 * np.pi)**3 * max(np.linalg.det(covariance), 10**-300))

  # Putting it all together, the exp portion of the equation divided by the pdf
  # np.sqrt(2 * np.pi)** 3 *, represents sqrt(2pi)^3 part of the equation
  # np.linalg.det(covariance) represents |Sigma| which is the determiant of the cov. matrix

  # print(pdf)

  # TA HINT to fix singular matrix:
  exponent = -0.5 * (diff.transpose() @ np.linalg.inv(covariance) @ diff)
  exponent = -np.log(-exponent)
  numerator = (np.exp(exponent)).item()
  denominator = np.sqrt(((2 * np.pi)**3) * np.linalg.det(covariance))

  pdf = numerator / denominator

    #did something wrong to cause singular matrix (something in E-M algorithm)

    #the errors we saw underflow, overflow, singular matrix, etc is b/c we did expectation maximization algorithm wrong

    # pay special attention to anything going into np.linalg.det(cov)
      #so that means pay special attention to covariance matrix




  return pdf


# TODO: Set parameters (threshold, prior)
threshhold = 0.00000000000001


#P(Orange)
prior = 0.000000001


# TODO: Send test images into algorithm to detect orange ball
test_path = "/content/test_images"
test_files = "*.jpg"
test_file_path = glob.glob(os.path.join(test_path, test_files))

outputs = []

i = 0
for x in test_file_path:
  image = cv2.imread(x)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

  originalImage = image.copy()

  for row in image:
    for pixel in row:

      #pass pixel into PDF to get P(x|Orange)
      likelihood = pdf(pixel, mean, covariance)

      #assumption: P(Orange|x) is proportional to P(x|Orange)*P(Orange)
      posterior = (likelihood * prior)

      if posterior >= threshhold:
        pixel[0] = 255
        pixel[1] = 255
        pixel[2] = 255
      else:
        pixel[0] = 0
        pixel[1] = 0
        pixel[2] = 0

  outputs.append((originalImage, image))
  # cv2.imwrite(f"singleGaussianOutput_{x.replace('/', '_')}.jpg", image)
  # print(x)
  # i += 1


  # plt.imshow(image)
  # plt.axis("off")
  # plt.show()


for original, result in outputs:
  fig, axes = plt.subplots(1, 2, figsize=(24, 8))

  original_rbg = cv2.cvtColor(original, cv2.COLOR_HSV2BGR)
  # result_rbg = cv2.cvtColor(result, cv2.COLOR_HSV2RGB)
  #axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_HSV2RGB))
  axes[0].imshow(original_rbg)
  axes[0].set_title('Original')

  axes[1].imshow(result)
  axes[1].set_title('Model Output')

  plt.tight_layout()
  plt.show()
```

## Project 2: Panorama Stitching

```python
# Download training images from Google Drive
import gdown
gdown.download_folder(id="1VAB_BG2gntlkwR059zR_8gd9pXajzgIk", quiet=True, use_cookies=False)

# Check whether the training images were successfully imported
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

train_image = mpimg.imread('/content/train_images/Set1/1.jpg')
plt.imshow(train_image)
plt.axis("off")
plt.show()

from google.colab import drive
drive.mount('/content/drive')

### Corner Detection
# 1) Convert image to gray scale image
# 2) Run harris or other corner detection from cv2 (cv2.cornerHarris OR cv2.goodFeaturesToTrack, etc.)
# Show the corner detection results for one image!!!


import cv2

def detect_corner(img):

  #convert to gray scale
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #extract corners w/ qualty
  corners = []
  dst = cv2.cornerHarris(img,2,3,0.04)
  largest = dst.max()
  for i in range(dst.shape[0]):
    for j in range(dst.shape[1]):
      if dst[i,j] > 0.01*largest:
        corners.append((j,i))


  return np.array(corners),dst

  # Show you result here


img = cv2.imread("/content/train_images/Set2/1.jpg")
corners,_ = detect_corner(img)

for (x,y) in corners:

  cv2.circle(img,(x,y),3,(0,0,255),-1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

### Adaptive Non-Maximal Suppression (or ANMS)
# Perform ANMS: Adaptive Non-Maximal Suppression
# Show ANMS output as an image
from scipy import ndimage
from skimage.feature import peak_local_max

#ASSUMPTIONS:
  #cmap is output from cv2.cornerHarris
  #ANMS is implemented completely seperate from part 1
def ANMS(cmap, corners, num_best):

  coordinates = corners

  radius = np.full(shape=(coordinates.shape[0],1), fill_value=999999999999999)

  coordinates = coordinates.T
  Xarray = coordinates[1]
  Yarray = coordinates[0]

  for i in range(len(Xarray)):
    for j in range(len(Xarray)):

      if (cmap[Xarray[j], Yarray[j]] > cmap[Xarray[i],Yarray[i]]):

        ED = (Xarray[j] - Xarray[i])**2 + (Yarray[j] - Yarray[i])**2
        if ED < radius[i]:
          radius[i] = ED

  #get sorted indicies
  sorted_indices = np.argsort(-radius, axis=0).reshape((1,-1))[0]

  out = np.zeros(shape=(num_best,2), dtype=int)

  #grab NStrongest
  for i in range(0, num_best):
    out[i] = ((Xarray[sorted_indices[i]], Yarray[sorted_indices[i]]))

  return out

# Show you result here

#no ANMS
img = cv2.imread("/content/train_images/Set2/1.jpg")
corners,_ = detect_corner(img)
for (x,y) in corners:
  cv2.circle(img,(x,y),3,(0,0,255),-1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()


#w/ ANMS
img = cv2.imread("/content/train_images/Set2/1.jpg")
corners, cmap = detect_corner(img)
corners = ANMS(cmap, corners, 30)
for (x,y) in corners:
  cv2.circle(img,(y,x),3,(0,0,255),-1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

def feature_descript(gray_img, corners):

  res = []
  for x,y in corners:

    start_row = x - 20
    end_row = x + 20
    if start_row < 0:
      start_row += -start_row
      end_row += -start_row
    if end_row > gray_img.shape[0]-1:
      end_row -= (end_row - gray_img.shape[0]-1)
      start_row -= (end_row - gray_img.shape[0]-1)


    start_col = y - 20
    end_col = y + 20
    if start_col < 0:
      start_col += -start_col
      end_col += -start_col
    if end_col > gray_img.shape[1]-1:
      end_col -= (end_col - gray_img.shape[1]-1)
      start_col -= (end_col - gray_img.shape[1]-1)


    patch = gray_img[start_row:end_row,start_col:end_col]
    blurred_patch = cv2.GaussianBlur(patch,(3,3),0)
    sub_sampled = cv2.resize(blurred_patch, (8,8), interpolation=cv2.INTER_AREA);

    reshaped = sub_sampled.astype(int).reshape((64,-1))
    final = (reshaped - np.mean(reshaped)) / np.std(reshaped)
    res.append((final, (x, y)))


  return res

  # Show you result here

#show corners we are aiming for
img = cv2.imread("/content/train_images/Set2/1.jpg")
corners, cmap = detect_corner(img)
corners = ANMS(cmap, corners, 30)
for x,y in corners:
  cv2.circle(img,(y,x),3,(0,0,255),-1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

#show feature descriptors
img = cv2.imread("/content/train_images/Set2/1.jpg")
descriptors = feature_descript(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), corners)
for item,_ in descriptors:
  plt.imshow(item.reshape((8,8)), cmap='gray')
  plt.axis("off")
  plt.show()


### Feature matching

#Features1 is the output from calling feature_descript() on the corners in img1
  #so it consits of (featureVector, location)
    # where location is a tuple of form (x,y)

#high level overview:
  # find differences[i][j]
  # for each corner[i] in image1 only match it w/ sortedSSDs[i][j] if there is a significant difference between sortedSSDs[i][j] and sortedSSDs[i][j+1]
    #this makes sure we only grab correspondences that have a high degree of certainty
def feature_match(img1, img2, features1, features2):

  #differences[i][j] = difference between features1[i] and features2[j]
  differences = np.zeros( (len(features1), len(features2)) )

#compute SSD for all points
  #for each feature in img1
  for i, (featureVector1, _) in enumerate(features1):
    #for each feature in img2
    for j, (featureVector2, _) in enumerate(features2):
      #for the 64 pixels in each feature
      for k in range(64):
        differences[i,j] += (featureVector1[k][0] - featureVector2[k][0])**2

  #list of cv2.DMatch objects
  dmatches = []

  ratio = 0.66
#find good matches
  for i in range(differences.shape[0]):
      #obtain argsort of differences[i]
      argsort = np.argsort(differences[i])

      #for all other differences conditionally append them for matches for i
      for index in range(len(argsort) -1):
        if differences[i,argsort[index]]/differences[i, argsort[index+1]] < ratio:
          dmatches.append(cv2.DMatch(_queryIdx=i, _trainIdx=argsort[index], _distance=differences[i,argsort[index]]))

        else:
          break


  return dmatches

  # Show you result here

#get feature descriptors for corners in image 1
img1 = cv2.imread("/content/train_images/Set2/2.jpg")
initial_corners1,cmap1 = detect_corner(img1)
corners1 = ANMS(cmap1, initial_corners1, 100)
features1 = feature_descript(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), corners1)

#get feature descriptors for corners in image 2
img2 = cv2.imread("/content/train_images/Set2/3.jpg")
initial_corners2,cmap2 = detect_corner(img2)
corners2 = ANMS(cmap2, initial_corners2, 100)
features2 = feature_descript(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), corners2)

#run feature_match
matches = feature_match(-1, -1, features1, features2)
print(f"len(matches) {len(matches)}")

#convert corners into key point objects
kp1 = []
for x,y in corners1:
  kp1.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

#convert corners into key point objects
kp2 = []
for x,y in corners2:
  kp2.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

#run drawMatches to create the final image
out = cv2.drawMatches(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), kp1, cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), kp2, matches, None)

plt.figure(figsize=(16, 12))
plt.imshow(out)
plt.axis("off")
plt.show()


# Ransac to filter out the wrong matchings and return homography
def RANSAC(match_kp1, match_kp2, N, t, threshold):
  max_inliers = []
  best_points1 = []
  best_points2 = []
  best_H = None
  k = 0

  while (k < N) and (len(max_inliers) < threshold):
    # print(f"i = {k} //// {len(max_inliers)} out of {threshold} inliers", flush=True)
    points1 = []
    points2 = []
    inliers = []

    # Choose four feature pairs at random
    ind = np.random.choice(len(match_kp1), size=4, replace=False)

    for i in ind:
      points1.append(match_kp1[i].pt)
      points2.append(match_kp2[i].pt)

    # Compute the homography matrix h using the selected points
    h = cv2.getPerspectiveTransform(np.float32(points1), np.float32(points2))


    # Compute inliers
    if h is not None:

      for j in range(len(match_kp1)):
        # Convert into homogeneous coordinates
        point1 = np.append(np.float32(match_kp1[j].pt), 1).reshape(3,1)
        # Compute Hpi
        projected_point = np.dot(h, point1)
        # Avoiding divide by zero
        if projected_point[2] > 0:
          # Convert into Cartesian coordinates
          projected_point = projected_point / projected_point[2]
          projected_point = projected_point[:2].reshape(-1)
          # Compute ssd
          ssd = np.sum((match_kp2[j].pt - projected_point)**2)
          # Add inliers
          if ssd < t:
            inliers.append((j,ssd))

    # Check if current inlier count is the largest so far
    if len(inliers) > len(max_inliers):
      max_inliers = inliers

    k = k + 1

  # Compute homography on 4 edges with least SSD (According to TA)
  sorted_inliers = sorted(max_inliers, key=lambda x: x[1])

  for i,sdd in sorted_inliers[:4]:
    best_points1.append(match_kp1[i].pt)
    best_points2.append(match_kp2[i].pt)

  best_H = cv2.getPerspectiveTransform(np.float32(best_points1),
                                       np.float32(best_points2))


  return best_H, max_inliers

  #get feature descriptors for corners in image 1
img1 = cv2.imread("/content/train_images/Set1/2.jpg")
initial_corners1,cmap1 = detect_corner(img1)
corners1 = ANMS(cmap1, initial_corners1, 100)
features1 = feature_descript(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), corners1)

#get feature descriptors for corners in image 2
img2 = cv2.imread("/content/train_images/Set1/3.jpg")
initial_corners2,cmap2 = detect_corner(img2)
corners2 = ANMS(cmap2, initial_corners2, 100)
features2 = feature_descript(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), corners2)

#run feature_match
matches = feature_match(-1, -1, features1, features2)

#convert corners into key point objects
kp1 = []
for x,y in corners1:
  kp1.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

#convert corners into key point objects
kp2 = []
for x,y in corners2:
  kp2.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

# Extract the matched points
match_kp1 = []
match_kp2 = []
for m in matches:
  match_kp1.append(kp1[m.queryIdx])
  match_kp2.append(kp2[m.trainIdx])

# RANSAC
N = 1000
t = 7
threshold = int(np.ceil(len(matches) * 0.9))  # 90% inliers threshold
best_h, inliers = RANSAC(match_kp1, match_kp2, N, t, threshold)

# Show you result here

# Draw only inlier matches
inlier_matches = [matches[i] for i,ssd in inliers]
print(f"{len(inlier_matches)} inliers")
# Visualize the inliers
out = cv2.drawMatches(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), kp1, cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), kp2, inlier_matches, None)


plt.figure(figsize=(16, 12))
plt.imshow(out)
plt.axis("off")
plt.show()

# NEW WARPP BLENDDDD
def warp_and_blend(img1, img2, inv_h):

  # Get the shape of the images
  h1, w1 = img1.shape[:2]  # Size of img1
  h2, w2 = img2.shape[:2]  # Size of img2

  # Define corners of img2
  corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)

  # Warp the corners of img2 using the homography
  warped_corners_img2 = cv2.perspectiveTransform(corners_img2, inv_h)

  # Define the corners of img1 (which is at (0,0) in the panorama space)
  corners_img1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32).reshape(-1, 1, 2)

  # Combine all corners (from warped img2 and img1) to calculate the bounding box
  all_corners = np.vstack((warped_corners_img2, corners_img1))

  # Find the bounding box for the panorama
  x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
  x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

  # Calculate the size of the output panorama
  output_width = x_max - x_min
  output_height = y_max - y_min

  # Create a translation matrix to move everything to the positive quadrant (if x_min or y_min < 0)
  translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

  # Warp img2 to the new size and translation
  warped_img = cv2.warpPerspective(img2, translation_matrix @ inv_h, (output_width, output_height))

  # Create a mask for img2
  mask = np.zeros(warped_img.shape, dtype=np.uint8)
  mask[warped_img > 0] = 255
  mask = mask[:, :, 0]

  # Enlarge img1 by padding with zeros (black) to match the size of the panorama
  img1_padded = np.zeros((output_height, output_width, 3), dtype=img1.dtype)
  img1_padded[-y_min:h1 - y_min, -x_min:w1 - x_min] = img1

  # Create a mask for img1 that identifies non-black pixels (valid region in warped_img2)
  mask_img2 = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
  _, mask_img2 = cv2.threshold(mask_img2, 1, 255, cv2.THRESH_BINARY)

  # Create a mask for img2 that identifies non-black pixels (valid region in img2_padded)
  mask_img1 = cv2.cvtColor(img1_padded, cv2.COLOR_BGR2GRAY)
  _, mask_img1 = cv2.threshold(mask_img1, 1, 255, cv2.THRESH_BINARY)

  # Combine the two masks to find the overlapping region (where both masks are 255)
  overlap_mask = cv2.bitwise_and(mask_img1, mask_img2)
  plt.imshow(overlap_mask)
  plt.show()

  # Find the non-overlapping region of img2 using a mask
  non_overlap_mask = cv2.bitwise_xor(mask_img2, overlap_mask)
  plt.imshow(non_overlap_mask)
  plt.show()

  # Add the non-overlapping region of img2 to img1_padded
  non_overlap_region = cv2.bitwise_and(warped_img, warped_img, mask=non_overlap_mask)
  img1_padded = cv2.add(img1_padded, non_overlap_region)
  plt.imshow(img1_padded)
  plt.show()

  # Find the center point for seamlessClone (center of the warped image)
  center_x = int((warped_corners_img2[:, :, 0].min() + warped_corners_img2[:, :, 0].max()) / 2 - x_min)
  center_y = int((warped_corners_img2[:, :, 1].min() + warped_corners_img2[:, :, 1].max()) / 2 - y_min)
  center = (center_x, center_y)
  print(center)

  # Perform seamless cloning using NORMAL_CLONE
  blend_img = cv2.seamlessClone(warped_img, img1_padded, mask, center, cv2.NORMAL_CLONE)
  # blend_img = img1_padded



  return blend_img, translation_matrix


  #get feature descriptors for corners in image 1
img1 = cv2.imread("/content/train_images/Set1/1.jpg")
initial_corners1,cmap1 = detect_corner(img1)
corners1 = ANMS(cmap1, initial_corners1, 100)
features1 = feature_descript(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), corners1)

#get feature descriptors for corners in image 2
img2 = cv2.imread("/content/train_images/Set1/2.jpg")
initial_corners2,cmap2 = detect_corner(img2)
corners2 = ANMS(cmap2, initial_corners2, 100)
features2 = feature_descript(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), corners2)

#run feature_match
matches = feature_match(-1, -1, features1, features2)

#convert corners into key point objects
kp1 = []
for x,y in corners1:
  kp1.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

#convert corners into key point objects
kp2 = []
for x,y in corners2:
  kp2.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

# Extract the matched points
match_kp1 = []
match_kp2 = []
for m in matches:
  match_kp1.append(kp1[m.queryIdx])
  match_kp2.append(kp2[m.trainIdx])

# RANSAC
N = 1000
t = 7
threshold = int(np.ceil(len(matches) * 0.9))  # 90% inliers threshold
best_h, inliers = RANSAC(match_kp1, match_kp2, N, t, threshold)
print(best_h)
# Show you result here
result = warp_and_blend(img1, img2, best_h)
# sharpen_kernel = np.array([[-1, -1, -1],
#                            [-1,  9, -1],
#                            [-1, -1, -1]])
# sharpened_image = cv2.filter2D(result, -1, sharpen_kernel)
# sharp_img = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)
# plt.imshow(sharp_img)
# plt.axis("off")
# plt.show()
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

plt.imshow(result)
plt.axis("off")
plt.show()
result2 = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
cv2.imwrite("result2.jpg", result2)

img2 = cv2.imread("/content/result2.jpg")
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.axis("off")
plt.show()

# MULTIPLYING HOMOGRAPHIES TEST

# COMPUTE HOMOGRAPHY OF IMG1 TO IMG2
#get feature descriptors for corners in image 1
img3 = cv2.imread("/content/train_images/Set1/1.jpg")
initial_corners3,cmap3 = detect_corner(img3)
corners3 = ANMS(cmap3, initial_corners3, 100)
features3 = feature_descript(cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY), corners3)

#get feature descriptors for corners in image 2
img4 = cv2.imread("/content/train_images/Set1/2.jpg")
initial_corners4,cmap4 = detect_corner(img4)
corners4 = ANMS(cmap4, initial_corners4, 100)
features4 = feature_descript(cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY), corners4)

#run feature_match
matches = feature_match(-1, -1, features3, features4)

#convert corners into key point objects
kp1 = []
for x,y in corners3:
  kp1.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

#convert corners into key point objects
kp2 = []
for x,y in corners4:
  kp2.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

# Extract the matched points
match_kp1 = []
match_kp2 = []
for m in matches:
  match_kp1.append(kp1[m.queryIdx])
  match_kp2.append(kp2[m.trainIdx])

# RANSAC
N = 1000
t = 7
threshold = int(np.ceil(len(matches) * 0.9))  # 90% inliers threshold
print(len(match_kp1))
h1to2, inliers = RANSAC(match_kp1, match_kp2, N, t, threshold)

# COMPUTE HOMOGRAPHY OF IMG2 TO IMG3
#get feature descriptors for corners in image 1
img3 = cv2.imread("/content/train_images/Set1/2.jpg")
initial_corners3,cmap3 = detect_corner(img3)
corners3 = ANMS(cmap3, initial_corners3, 100)
features3 = feature_descript(cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY), corners3)

#get feature descriptors for corners in image 2
img4 = cv2.imread("/content/train_images/Set1/3.jpg")
initial_corners4,cmap4 = detect_corner(img4)
corners4 = ANMS(cmap4, initial_corners4, 100)
features4 = feature_descript(cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY), corners4)

#run feature_match
matches = feature_match(-1, -1, features3, features4)

#convert corners into key point objects
kp1 = []
for x,y in corners3:
  kp1.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

#convert corners into key point objects
kp2 = []
for x,y in corners4:
  kp2.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

# Extract the matched points
match_kp1 = []
match_kp2 = []
for m in matches:
  match_kp1.append(kp1[m.queryIdx])
  match_kp2.append(kp2[m.trainIdx])

# RANSAC
N = 1000
t = 7
threshold = int(np.ceil(len(matches) * 0.9))  # 90% inliers threshold
print(len(match_kp1))
h2to3, inliers = RANSAC(match_kp1, match_kp2, N, t, threshold)





# WARP AND BLEND IMG2 INTO IMG1
img1 = cv2.imread("/content/train_images/Set1/1.jpg")
img2 = cv2.imread("/content/train_images/Set1/2.jpg")

# inv_best_h = np.linalg.inv(best_h) # 2 to 1
# inv_h2to3 = np.linalg.inv(h2to3) # 3 to 2
# inv_h = np.linalg.inv(best_h @ h2to3) # 3 to 1
inv_h = np.linalg.inv(h1to2) # Inverse to get hg from img2 to img1
print(f"FIRST inv_h: {inv_h}")

# Get the shape of the images
h1, w1 = img1.shape[:2]  # Size of img1
h2, w2 = img2.shape[:2]  # Size of img2

# Define corners of img2
corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)

# Warp the corners of img2 using the homography
warped_corners_img2 = cv2.perspectiveTransform(corners_img2, inv_h)

# Define the corners of img1 (which is at (0,0) in the panorama space)
corners_img1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32).reshape(-1, 1, 2)

# Combine all corners (from warped img2 and img1) to calculate the bounding box
all_corners = np.vstack((warped_corners_img2, corners_img1))

# Find the bounding box for the panorama
x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

# Calculate the size of the output panorama
output_width = x_max - x_min
output_height = y_max - y_min

# Create a translation matrix to move everything to the positive quadrant (if x_min or y_min < 0)
translation_matrix1 = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
print(f"FIRST T: {translation_matrix1}")

# Warp img2 to the new size and translation
warped_img = cv2.warpPerspective(img2, translation_matrix1 @ inv_h, (output_width, output_height))

# Create a mask for img2
mask = np.zeros(warped_img.shape, dtype=np.uint8)
mask[warped_img > 0] = 255
mask = mask[:, :, 0]

# Enlarge img1 by padding with zeros (black) to match the size of the panorama
img1_padded = np.zeros((output_height, output_width, 3), dtype=img1.dtype)
img1_padded[-y_min:h1 - y_min, -x_min:w1 - x_min] = img1

# Create a mask for img1 that identifies non-black pixels (valid region in warped_img2)
mask_img2 = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
_, mask_img2 = cv2.threshold(mask_img2, 1, 255, cv2.THRESH_BINARY)

# Create a mask for img2 that identifies non-black pixels (valid region in img2_padded)
mask_img1 = cv2.cvtColor(img1_padded, cv2.COLOR_BGR2GRAY)
_, mask_img1 = cv2.threshold(mask_img1, 1, 255, cv2.THRESH_BINARY)

# Combine the two masks to find the overlapping region (where both masks are 255)
overlap_mask = cv2.bitwise_and(mask_img1, mask_img2)
plt.imshow(overlap_mask)
plt.show()

# Find the non-overlapping region of img2 using a mask
non_overlap_mask = cv2.bitwise_xor(mask_img2, overlap_mask)
plt.imshow(non_overlap_mask)
plt.show()

# Add the non-overlapping region of img2 to img1_padded
non_overlap_region = cv2.bitwise_and(warped_img, warped_img, mask=non_overlap_mask)
img1_padded = cv2.add(img1_padded, non_overlap_region)
plt.imshow(img1_padded)
plt.show()

# Find the center point for seamlessClone (center of the warped image)
center_x = int((warped_corners_img2[:, :, 0].min() + warped_corners_img2[:, :, 0].max()) / 2 - x_min)
center_y = int((warped_corners_img2[:, :, 1].min() + warped_corners_img2[:, :, 1].max()) / 2 - y_min)
center = (center_x, center_y)
print(center)

# Perform seamless cloning using NORMAL_CLONE
blend_img = cv2.seamlessClone(warped_img, img1_padded, mask, center, cv2.NORMAL_CLONE)

plt.imshow(blend_img)
plt.axis("off")
plt.show()






# WARP AND BLEND IMG3 INTO IMG1AND2
img1 = blend_img
img2 = img4

inv_best_h = np.linalg.inv(h1to2) # 2 to 1
inv_h2to3 = np.linalg.inv(h2to3) # 3 to 2
inv_h = inv_best_h @ inv_h2to3 # 3 to 1
inv_h = translation_matrix1 @ inv_h
print(f"SECOND inv_h: {inv_h}")
# inv_h = np.linalg.inv(H) # Inverse to get hg from img2 to img1

# Get the shape of the images
h1, w1 = img1.shape[:2]  # Size of img1
h2, w2 = img2.shape[:2]  # Size of img2

# Define corners of img2
corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)

# Warp the corners of img2 using the homography
warped_corners_img2 = cv2.perspectiveTransform(corners_img2, inv_h)

# Define the corners of img1 (which is at (0,0) in the panorama space)
corners_img1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32).reshape(-1, 1, 2)

# Combine all corners (from warped img2 and img1) to calculate the bounding box
all_corners = np.vstack((warped_corners_img2, corners_img1))

# Find the bounding box for the panorama
x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

# Calculate the size of the output panorama
output_width = x_max - x_min
output_height = y_max - y_min

# Create a translation matrix to move everything to the positive quadrant (if x_min or y_min < 0)
translation_matrix2 = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

# Warp img2 to the new size and translation
warped_img = cv2.warpPerspective(img2, translation_matrix2 @ inv_h, (output_width, output_height))

# Create a mask for img2
mask = np.zeros(warped_img.shape, dtype=np.uint8)
mask[warped_img > 0] = 255
mask = mask[:, :, 0]

# Enlarge img1 by padding with zeros (black) to match the size of the panorama
img1_padded = np.zeros((output_height, output_width, 3), dtype=img1.dtype)
img1_padded[-y_min:h1 - y_min, -x_min:w1 - x_min] = img1

# Create a mask for img1 that identifies non-black pixels (valid region in warped_img2)
mask_img2 = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
_, mask_img2 = cv2.threshold(mask_img2, 1, 255, cv2.THRESH_BINARY)

# Create a mask for img2 that identifies non-black pixels (valid region in img2_padded)
mask_img1 = cv2.cvtColor(img1_padded, cv2.COLOR_BGR2GRAY)
_, mask_img1 = cv2.threshold(mask_img1, 1, 255, cv2.THRESH_BINARY)

# Combine the two masks to find the overlapping region (where both masks are 255)
overlap_mask = cv2.bitwise_and(mask_img1, mask_img2)
plt.imshow(overlap_mask)
plt.show()

# Find the non-overlapping region of img2 using a mask
non_overlap_mask = cv2.bitwise_xor(mask_img2, overlap_mask)
plt.imshow(non_overlap_mask)
plt.show()

# Add the non-overlapping region of img2 to img1_padded
non_overlap_region = cv2.bitwise_and(warped_img, warped_img, mask=non_overlap_mask)
img1_padded = cv2.add(img1_padded, non_overlap_region)
plt.imshow(img1_padded)
plt.show()

# Find the center point for seamlessClone (center of the warped image)
center_x = int((warped_corners_img2[:, :, 0].min() + warped_corners_img2[:, :, 0].max()) / 2 - x_min)
center_y = int((warped_corners_img2[:, :, 1].min() + warped_corners_img2[:, :, 1].max()) / 2 - y_min)
center = (center_x, center_y)
print(center)

# Perform seamless cloning using NORMAL_CLONE
blend_img = cv2.seamlessClone(warped_img, img1_padded, mask, center, cv2.NORMAL_CLONE)

plt.imshow(blend_img)
plt.axis("off")
plt.show()

#get feature descriptors for corners in image 1
img3 = cv2.imread("/content/result2.jpg")
initial_corners3,cmap3 = detect_corner(img3)
corners3 = ANMS(cmap3, initial_corners3, 100)
features3 = feature_descript(cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY), corners3)

#get feature descriptors for corners in image 2
img4 = cv2.imread("/content/train_images/Set1/1.jpg")
initial_corners4,cmap4 = detect_corner(img4)
corners4 = ANMS(cmap4, initial_corners4, 100)
features4 = feature_descript(cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY), corners4)

#run feature_match
matches = feature_match(-1, -1, features3, features4)

#convert corners into key point objects
kp1 = []
for x,y in corners3:
  kp1.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

#convert corners into key point objects
kp2 = []
for x,y in corners4:
  kp2.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

# Extract the matched points
match_kp1 = []
match_kp2 = []
for m in matches:
  match_kp1.append(kp1[m.queryIdx])
  match_kp2.append(kp2[m.trainIdx])

# RANSAC
N = 1000
t = 7
threshold = int(np.ceil(len(matches) * 0.9))  # 90% inliers threshold
print(len(match_kp1))
best_h, inliers = RANSAC(match_kp1, match_kp2, N, t, threshold)

# Show you result here
result = warp_and_blend(img3, img4, best_h)
plt.imshow(result)
plt.axis("off")
plt.show()

#get feature descriptors for corners in image 1
img1 = cv2.imread("/content/train_images/Set2/2.jpg")
initial_corners1,cmap1 = detect_corner(img1)
corners1 = ANMS(cmap1, initial_corners1, 100)
features1 = feature_descript(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), corners1)

#get feature descriptors for corners in image 2
img2 = cv2.imread("/content/train_images/Set2/2.jpg")
initial_corners2,cmap2 = detect_corner(img2)
corners2 = ANMS(cmap2, initial_corners2, 100)
features2 = feature_descript(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), corners2)

#run feature_match
matches = feature_match(-1, -1, features1, features2)

#convert corners into key point objects
kp1 = []
for x,y in corners1:
  kp1.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

#convert corners into key point objects
kp2 = []
for x,y in corners2:
  kp2.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

# Extract the matched points
match_kp1 = []
match_kp2 = []
for m in matches:
  match_kp1.append(kp1[m.queryIdx])
  match_kp2.append(kp2[m.trainIdx])

# RANSAC
N = 1000
t = 7
threshold = int(np.ceil(len(matches) * 0.9))  # 90% inliers threshold
best_h, inliers = RANSAC(match_kp1, match_kp2, N, t, threshold)

# Show you result here

# Draw only inlier matches
inlier_matches = [matches[i] for i,ssd in inliers]

# Visualize the inliers
out = cv2.drawMatches(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), kp1, cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), kp2, inlier_matches, None)


plt.figure(figsize=(16, 12))
plt.imshow(out)
plt.axis("off")
plt.show()

N = 1000
t = 7
threshold = int(np.ceil(len(matches) * 0.9))  # 90% inliers threshold
best_h, inliers = RANSAC(match_kp1, match_kp2, N, t, threshold)

# Show you result here
inv_h = np.linalg.inv(best_h) # Inverse to get hg from img2 to img1

# Get the shape of the images
h1, w1 = img1.shape[:2]  # Size of img1
h2, w2 = img2.shape[:2]  # Size of img2
print(h1, w1)
print(h2, w2)

# Define corners of img2
corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)
print(corners_img2)

# Warp the corners of img2 using the homography
warped_corners_img2 = cv2.perspectiveTransform(corners_img2, inv_h)

# Define the corners of img1 (which is at (0,0) in the panorama space)
corners_img1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32).reshape(-1, 1, 2)

# Combine all corners (from warped img2 and img1) to calculate the bounding box
all_corners = np.vstack((warped_corners_img2, corners_img1))

# Find the bounding box for the panorama
x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
print(x_min)
print(x_max)
print(y_min)
print(y_max)

# Calculate the size of the output panorama
output_width = x_max - x_min
output_height = y_max - y_min

# Create a translation matrix to move everything to the positive quadrant (if x_min or y_min < 0)
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

# Warp img2 to the new size and translation
warped_img = cv2.warpPerspective(img2, translation_matrix @ inv_h, (output_width, output_height))

# Create a mask for img (all white)
mask = np.zeros(warped_img.shape, dtype=np.uint8)
mask[warped_img > 0] = 255
mask = mask[:, :, 0]

# Enlarge img1 by padding with zeros (black) to match the size of the panorama
img1_padded = np.zeros((output_height, output_width, 3), dtype=img1.dtype)
img1_padded[-y_min:h1 - y_min, -x_min:w1 - x_min] = img1


# Find the center point for seamlessClone (center of the warped image)
center_x = int((warped_corners_img2[:, :, 0].min() + warped_corners_img2[:, :, 0].max()) / 2 - x_min)
center_y = int((warped_corners_img2[:, :, 1].min() + warped_corners_img2[:, :, 1].max()) / 2 - y_min)
center = (center_x, center_y)
print(center)

plt.imshow(img1_padded)
plt.show()
plt.imshow(warped_img)
plt.show()
plt.imshow(mask)
plt.show()
# Perform seamless cloning using NORMAL_CLONE
blend_img = cv2.seamlessClone(warped_img, img1_padded, mask, center, cv2.NORMAL_CLONE)

# plt.imshow(img1)
# plt.show()
# plt.imshow(img2)
# plt.show()
# plt.imshow(warped_img)
# plt.show()
plt.imshow(blend_img)
plt.show()

def pano_imgs(img_list):
  # Compute inverse homography for each image
  inv_H_list = []
  current_img = img_list[0]

  for next_img in img_list[1:]:
    #get feature descriptors for corners in current_img
    initial_corners1,cmap1 = detect_corner(current_img)
    corners1 = ANMS(cmap1, initial_corners1, 100)
    features1 = feature_descript(cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY),
                                 corners1)

    #get feature descriptors for corners in next img
    initial_corners2,cmap2 = detect_corner(next_img)
    corners2 = ANMS(cmap2, initial_corners2, 100)
    features2 = feature_descript(cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY),
                                 corners2)

    #run feature_match
    matches = feature_match(-1, -1, features1, features2)

    #convert corners into key point objects
    kp1 = []
    for x,y in corners1:
      kp1.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

    #convert corners into key point objects
    kp2 = []
    for x,y in corners2:
      kp2.append(cv2.KeyPoint(y.astype(float),x.astype(float),1))

    # Extract the matched points
    match_kp1 = []
    match_kp2 = []
    for m in matches:
      match_kp1.append(kp1[m.queryIdx])
      match_kp2.append(kp2[m.trainIdx])

    # RANSAC
    N = 1000
    t = 7
    threshold = int(np.ceil(len(matches) * 0.9))  # 90% inliers threshold
    best_h, inliers = RANSAC(match_kp1, match_kp2, N, t, threshold)


    inv_H_list.append(np.linalg.inv(best_h))
    current_img = next_img

  # Use inv_H_list to warp and blend
  main_img = img_list[0]
  translations = []

  for i in range(1, len(img_list)):
    current_h = np.eye(3)
    current_t = np.eye(3)

    # Multiply all previous homographies
    for j in range(i):
      # print(f"inv_H {j} BEFORE @ {inv_H_list[j]}")
      current_h = current_h @ inv_H_list[j]

    # Multiple all previous translations
    for k in translations:
      print(f"FIRST T: {k}")
      current_t = k @ current_t

    # Multiply translation into homography
    current_h = current_t @ current_h
    print(f"{i} inv_h: {current_h}")
    # print(f"inv_H AFTER @ {current_h}")
    # Warp and blend
    main_img, new_t = warp_and_blend(main_img, img_list[i], current_h)
    translations.append(new_t)
    plt.imshow(main_img)
    plt.show()

  return main_img

  import glob
import re

# Sort images to read them in order
def extract_number(path):
    # Extract the filename part and use regex to find the number in the image name
    match = re.search(r'(\d+)\.jpg', path)
    return int(match.group(1)) if match else 0

def middle_left_right_order(images):
    result = []
    n = len(images)
    middle = n // 2  # Find the index of the middle image

    # Add the middle image first
    result.append(images[middle])

    # Alternate between left and right of the middle
    left = middle - 1
    right = middle + 1

    while left >= 0 or right < n:
      if right < n:
            result.append(images[right])
            right += 1
      if left >= 0:
          result.append(images[left])
          left -= 1


    return result

# Show your result here
img_list = []
folder_path = "/content/train_images/Set1/"
image_paths = glob.glob(folder_path + "*.jpg")
image_paths = sorted(image_paths, key=extract_number)
# image_paths = middle_left_right_order(image_paths)

for path in image_paths:
  img_list.append(cv2.imread(path))
print(image_paths)

result = pano_imgs(img_list)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

img_list = []
folder_path = "/content/train_images/Set2/"
image_paths = glob.glob(folder_path + "*.jpg")
image_paths = sorted(image_paths, key=extract_number)
# image_paths = middle_left_right_order(image_paths)

for path in image_paths:
  img_list.append(cv2.imread(path))
print(image_paths)

result = pano_imgs(img_list)
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()

img_list = []
folder_path = "/content/train_images/Set3/"
image_paths = glob.glob(folder_path + "*.jpg")
image_paths = sorted(image_paths, key=extract_number)
# image_paths = middle_left_right_order(image_paths)

for path in image_paths:
  img_list.append(cv2.imread(path))
print(image_paths)

# max_count = 100
# count = 0
# while count < max_count:
#   try:
#     result = pano_imgs(img_list)
#   except:
#     count += 1
#     continue
result = pano_imgs(img_list)
# print(f"Retries: {count}")
result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
plt.imshow(result)
plt.axis("off")
plt.show()
```

## Project 3: From Pixels to 3D Worlds : Two-View 3D Reconstruction

## Project 4: Depth Estimation
