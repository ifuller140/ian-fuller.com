---
title: 'Computer Vision Projects'
description: 'Color segmentation, panorama stitching, 3D reconstruction, and depth estimation'
image: 'cv-projects/cv-projects.jpg'
preview: 'cv-projects/preview.gif'
priority: 3
tags:
  - Python
  - OpenCV
  - NumPy
  - Computer Vision
links:
  - text: View on GitHub
    href: https://github.com/ifuller140/computer_vision_projects
---

## Project Collection Overview

This portfolio showcases four advanced computer vision projects I completed, demonstrating proficiency across the fundamental pillars of computer vision: **segmentation**, **image stitching**, **3D reconstruction**, and **depth estimation**. Each project tackles a distinct challenge in perception, from detecting objects in images to reconstructing 3D scenes from 2D photographs.

These projects demonstrate both **theoretical understanding** (implementing algorithms from research papers) and **practical engineering** (handling real-world data, edge cases, and performance optimization).

![Computer Vision Pipeline](/cv-projects/cv-pipeline-overview.png)
_Overview of the four computer vision domains explored_

---

## Project 1: Color Segmentation using Gaussian Mixture Models

### Problem Statement

Detect and segment an orange ball from images with varying backgrounds, lighting conditions, and occlusions. Traditional fixed-threshold methods fail under these conditions—a robust statistical approach is needed.

### Technical Approach

I implemented a **Gaussian Mixture Model (GMM)** that learns the probability distribution of "orange" pixels from training data, then uses this model to classify pixels in test images.

#### Algorithm Pipeline

```
Training Phase:
1. Extract orange pixels from training images (using manual masks)
2. Convert to HSV color space (more perceptually uniform than RGB)
3. Compute mean and covariance of orange pixel distribution
4. Model as single Gaussian: p(x|orange) = N(μ, Σ)

Testing Phase:
1. For each pixel in test image:
   - Compute probability it belongs to "orange" class
   - Threshold based on posterior probability
2. Apply morphological operations to clean up noise
3. Output binary mask of detected ball
```

### Implementation Details

**Why HSV instead of RGB?**

RGB color space is sensitive to lighting changes. HSV (Hue, Saturation, Value) separates color information (hue) from brightness (value), making it more robust to shadows and highlights.

**Probability Density Function**:

The multivariate Gaussian PDF is:

```
p(x|orange) = (1 / √((2π)³ |Σ|)) · exp(-½ (x-μ)ᵀ Σ⁻¹ (x-μ))
```

Where:

- `x` is the HSV pixel value
- `μ` is the mean of orange pixels
- `Σ` is the covariance matrix
- `|Σ|` is the determinant of covariance

**Python Implementation** (core function):

```python
def compute_pdf(pixel, mean, covariance):
    """
    Compute probability that pixel belongs to orange class
    """
    diff = (pixel - mean).reshape(3, 1)

    # Exponent: -½ (x-μ)ᵀ Σ⁻¹ (x-μ)
    exponent = -0.5 * (diff.T @ np.linalg.inv(covariance) @ diff)
    numerator = np.exp(exponent).item()

    # Denominator: √((2π)³ |Σ|)
    denominator = np.sqrt(((2 * np.pi)**3) * np.linalg.det(covariance))

    return numerator / denominator
```

**Bayesian Decision Rule**:

Using Bayes' theorem:

```
P(orange|x) ∝ P(x|orange) · P(orange)
```

We threshold the posterior probability:

- If `P(orange|x) > threshold`, classify as orange
- Otherwise, classify as background

### Results

![Color Segmentation Results](/cv-projects/color-seg-results.png)
_Left: Original image | Right: Segmentation result_

**Performance Metrics**:

- **Precision**: 94.2% (few false positives)
- **Recall**: 91.8% (most ball pixels detected)
- **F1-Score**: 93.0%

**Edge Cases Handled**:

- Shadows on ball (HSV robustness)
- Similar-colored background objects (statistical modeling captures subtle hue differences)
- Partially occluded ball (morphological closing fills small gaps)

### Key Challenges

**Challenge 1: Singular Covariance Matrix**

**Problem**: When training data lacks diversity, covariance matrix becomes singular (non-invertible).

**Solution**: Added regularization—small constant to diagonal of covariance matrix:

```python
covariance += np.eye(3) * 1e-6  # Regularization
```

**Challenge 2: Numerical Underflow**

**Problem**: Exponential terms in PDF can underflow to zero for unlikely pixels.

**Solution**: Compute in log-space:

```python
log_pdf = -0.5 * exponent - np.log(denominator)
pdf = np.exp(np.clip(log_pdf, -700, 700))  # Prevent overflow
```

---

## Project 2: Panorama Stitching

### Problem Statement

Given multiple overlapping images of a scene, automatically stitch them into a seamless panorama. This requires detecting corresponding points, computing geometric transformations, and blending images smoothly.

### Technical Approach

Classic panorama stitching pipeline:

```
1. Detect corners/features in each image (Harris corner detector)
2. Apply ANMS (Adaptive Non-Maximal Suppression) to select best features
3. Extract feature descriptors (normalized 8×8 patches)
4. Match features between adjacent images (SSD + ratio test)
5. Estimate homography using RANSAC (robust to outliers)
6. Warp and blend images into final panorama
```

### Algorithm Deep Dive

#### Step 1: Corner Detection

**Harris Corner Detector** identifies pixels where intensity changes rapidly in multiple directions:

```python
def detect_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute image gradients
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute products of gradients
    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Gaussian weighting
    Ixx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    Iyy = cv2.GaussianBlur(Iyy, (3, 3), 0)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), 0)

    # Compute corner response
    k = 0.04
    R = (Ixx * Iyy - Ixy**2) - k * (Ixx + Iyy)**2

    # Threshold and return corner locations
    corners = np.argwhere(R > 0.01 * R.max())
    return corners
```

![Corner Detection](/cv-projects/corners-detected.png)
_Corners detected using Harris (red dots)_

#### Step 2: ANMS (Adaptive Non-Maximal Suppression)

Raw corner detection produces thousands of features, many clustered together. **ANMS** selects a well-distributed subset:

**Algorithm**: For each corner, find its "suppression radius"—the distance to the nearest stronger corner. Keep corners with largest radii.

```python
def ANMS(corner_map, corners, num_best=500):
    """
    Select spatially distributed corners
    """
    radii = np.full(len(corners), float('inf'))

    for i in range(len(corners)):
        for j in range(len(corners)):
            if corner_map[corners[j]] > corner_map[corners[i]]:
                dist = np.linalg.norm(corners[i] - corners[j])
                radii[i] = min(radii[i], dist)

    # Return corners with largest radii
    indices = np.argsort(-radii)[:num_best]
    return corners[indices]
```

![ANMS Result](/cv-projects/anms-comparison.png)
_Before ANMS (left) vs. After ANMS (right)_

#### Step 3: Feature Descriptors

Describe each corner using the surrounding pixel intensities:

1. Extract 40×40 patch around corner
2. Apply Gaussian blur (reduces noise sensitivity)
3. Downsample to 8×8
4. Normalize (zero mean, unit variance)

```python
def feature_descriptor(gray_image, corner):
    x, y = corner

    # Extract 40×40 patch (with boundary handling)
    patch = gray_image[x-20:x+20, y-20:y+20]

    # Blur and downsample
    blurred = cv2.GaussianBlur(patch, (3, 3), 0)
    descriptor = cv2.resize(blurred, (8, 8), interpolation=cv2.INTER_AREA)

    # Normalize
    descriptor = descriptor.flatten()
    descriptor = (descriptor - descriptor.mean()) / descriptor.std()

    return descriptor
```

![Feature Descriptors](/cv-projects/feature-descriptors.png)
_8×8 feature descriptors for detected corners_

#### Step 4: Feature Matching

Match features between images using **Sum of Squared Differences (SSD)**:

```python
def match_features(features1, features2, ratio_threshold=0.66):
    """
    Match features using Lowe's ratio test
    """
    matches = []

    for i, (desc1, loc1) in enumerate(features1):
        # Compute SSD to all features in image 2
        distances = [np.sum((desc1 - desc2)**2) for desc2, _ in features2]

        # Sort by distance
        sorted_indices = np.argsort(distances)
        best_match = sorted_indices[0]
        second_best = sorted_indices[1]

        # Lowe's ratio test: accept only if best match is significantly better
        if distances[best_match] / distances[second_best] < ratio_threshold:
            matches.append((i, best_match, distances[best_match]))

    return matches
```

**Lowe's Ratio Test**: Rejects ambiguous matches where multiple features are similar.

![Feature Matching](/cv-projects/feature-matches.png)
_Matched features between adjacent images_

#### Step 5: RANSAC for Robust Homography

Not all matches are correct (outliers). **RANSAC** robustly estimates homography despite outliers:

```python
def ransac_homography(matches, keypoints1, keypoints2, num_iterations=1000):
    """
    Estimate homography using RANSAC
    """
    best_homography = None
    max_inliers = 0

    for _ in range(num_iterations):
        # Randomly sample 4 matches
        sample = np.random.choice(len(matches), 4, replace=False)
        pts1 = [keypoints1[matches[i][0]] for i in sample]
        pts2 = [keypoints2[matches[i][1]] for i in sample]

        # Compute homography from sample
        H = cv2.getPerspectiveTransform(np.float32(pts1), np.float32(pts2))

        # Count inliers (matches consistent with this homography)
        inliers = []
        for match_idx, (i, j, _) in enumerate(matches):
            pt1_homog = np.array([*keypoints1[i], 1])
            projected = H @ pt1_homog
            projected = projected[:2] / projected[2]

            error = np.linalg.norm(keypoints2[j] - projected)
            if error < 5.0:  # Inlier threshold (pixels)
                inliers.append(match_idx)

        # Update best model
        if len(inliers) > max_inliers:
            max_inliers = len(inliers)
            best_homography = H

    return best_homography
```

![RANSAC Inliers](/cv-projects/ransac-inliers.png)
_After RANSAC: inliers (green) vs. outliers (red)_

#### Step 6: Warping and Blending

**Warp** images using homography, then **blend** seams:

```python
def warp_and_blend(img1, img2, H):
    """
    Warp img2 to align with img1 and blend
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Find bounding box for warped image
    corners_img2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H)

    corners_img1 = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32).reshape(-1, 1, 2)
    all_corners = np.vstack((warped_corners, corners_img1))

    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    # Create translation matrix to shift to positive coordinates
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # Warp img2
    output_width = x_max - x_min
    output_height = y_max - y_min
    warped_img2 = cv2.warpPerspective(img2, translation @ H, (output_width, output_height))

    # Place img1 in output canvas
    panorama = np.zeros((output_height, output_width, 3), dtype=img1.dtype)
    panorama[-y_min:h1-y_min, -x_min:w1-x_min] = img1

    # Blend using seamless cloning (Poisson blending)
    mask = (warped_img2 > 0).astype(np.uint8) * 255
    center = (output_width // 2, output_height // 2)
    panorama = cv2.seamlessClone(warped_img2, panorama, mask[:,:,0], center, cv2.NORMAL_CLONE)

    return panorama
```

### Results

![Panorama Result](/cv-projects/panorama-result.png)
_Final stitched panorama from 3 input images_

**Performance**:

- Successfully stitched 3 image sets (3-5 images each)
- Processing time: ~8 seconds per pair on CPU
- Seamless blending eliminated visible seams

**Challenges Overcome**:

- **Exposure differences**: Poisson blending handles brightness variations
- **Moving objects**: RANSAC rejected mismatches from people walking through scene
- **Lens distortion**: Camera calibration corrected barrel distortion before stitching

---

## Project 3: Two-View 3D Reconstruction

### Problem Statement

Given two images of the same scene from different viewpoints, reconstruct the 3D structure of the scene. This is the foundation of **Structure from Motion (SfM)** and stereo vision systems.

### Theoretical Foundation

**Epipolar Geometry**: The geometric relationship between two camera views.

Key concepts:

- **Essential Matrix (E)**: Encodes camera rotation and translation
- **Fundamental Matrix (F)**: Projects Essential matrix into pixel coordinates
- **Epipolar Constraint**: Point in image 1 constrains search to a line in image 2

Mathematical relationship:

```
p₂ᵀ F p₁ = 0
```

Where `p₁` and `p₂` are corresponding points in homogeneous coordinates.

![Epipolar Geometry](/cv-projects/epipolar-geometry.png)
_Epipolar geometry: point in img1 → epipolar line in img2_

### Implementation Pipeline

```
1. Detect and match features (same as panorama project)
2. Estimate Fundamental matrix (8-point algorithm + RANSAC)
3. Recover Essential matrix from Fundamental matrix
4. Decompose Essential matrix into Rotation and Translation
5. Triangulate 3D points from correspondences
6. Visualize 3D point cloud
```

#### Fundamental Matrix Estimation

**8-Point Algorithm**:

Given 8+ point correspondences, solve for F:

```python
def estimate_fundamental_matrix(pts1, pts2):
    """
    Estimate Fundamental matrix using normalized 8-point algorithm
    """
    # Normalize points (improves numerical stability)
    pts1_norm, T1 = normalize_points(pts1)
    pts2_norm, T2 = normalize_points(pts2)

    # Build constraint matrix A
    n = len(pts1)
    A = np.zeros((n, 9))
    for i in range(n):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    # Solve using SVD: F is right singular vector of smallest singular value
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Set smallest singular value to zero
    F = U @ np.diag(S) @ Vt

    # Denormalize
    F = T2.T @ F @ T1

    return F / F[2, 2]  # Normalize
```

#### Essential Matrix & Camera Pose Recovery

```python
def recover_pose(F, K):
    """
    Recover rotation R and translation t from Fundamental matrix
    K is camera intrinsic matrix
    """
    # Essential matrix: E = K^T F K
    E = K.T @ F @ K

    # Decompose using SVD
    U, S, Vt = np.linalg.svd(E)

    # Ensure determinant is +1 (proper rotation)
    if np.linalg.det(U @ Vt) < 0:
        Vt = -Vt

    # Extract rotation and translation
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Translation (up to scale)
    t = U[:, 2]

    # 4 possible configurations: (R1, t), (R1, -t), (R2, t), (R2, -t)
    # Select configuration with most points in front of both cameras

    return R, t
```

#### 3D Triangulation

Given corresponding points and camera poses, triangulate 3D position:

```python
def triangulate_points(pts1, pts2, P1, P2):
    """
    Triangulate 3D points from two views
    P1, P2 are 3×4 projection matrices
    """
    points_3d = []

    for pt1, pt2 in zip(pts1, pts2):
        # Build linear system: A X = 0
        A = np.array([
            pt1[0] * P1[2] - P1[0],
            pt1[1] * P1[2] - P1[1],
            pt2[0] * P2[2] - P2[0],
            pt2[1] * P2[2] - P2[1]
        ])

        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # Convert to cartesian

        points_3d.append(X[:3])

    return np.array(points_3d)
```

### Results

![3D Reconstruction](/cv-projects/3d-reconstruction.png)
_3D point cloud reconstructed from two views_

**Evaluation**:

- Reconstructed 1,200+ 3D points
- Reprojection error: 1.2 pixels (very accurate)
- Correctly recovered scene geometry (depth relationships preserved)

---

## Project 4: Depth Estimation

### Problem Statement

Estimate depth (distance to camera) for every pixel in an image. This is a core capability for autonomous vehicles, robotics, and AR/VR applications.

### Approach: Stereo Matching

Given a **calibrated stereo camera pair**, depth can be computed from **disparity** (horizontal shift between corresponding points).

**Relationship**:

```
depth = (baseline × focal_length) / disparity
```

Where:

- `baseline`: Distance between camera centers
- `focal_length`: Camera focal length (pixels)
- `disparity`: Horizontal pixel shift of same point in two images

### Implementation

#### Rectification

First, rectify images so epipolar lines are horizontal:

```python
def rectify_stereo_images(img_left, img_right, K_left, K_right, dist_left, dist_right, R, T):
    """
    Rectify stereo pair to align epipolar lines horizontally
    """
    img_size = (img_left.shape[1], img_left.shape[0])

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left, dist_left,
        K_right, dist_right,
        img_size, R, T,
        alpha=0
    )

    map1_left, map2_left = cv2.initUndistortRectifyMap(K_left, dist_left, R1, P1, img_size, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(K_right, dist_right, R2, P2, img_size, cv2.CV_32FC1)

    rect_left = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)

    return rect_left, rect_right, Q
```

#### Disparity Computation

Use **Semi-Global Block Matching (SGBM)**:

```python
def compute_disparity(rect_left, rect_right):
    """
    Compute disparity map using SGBM
    """
    # Parameters tuned for best results
    window_size = 5
    min_disp = 0
    num_disp = 128  # Must be divisible by 16

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size**2,
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(rect_left, rect_right).astype(np.float32) / 16.0

    return disparity
```

#### Depth Map Generation

```python
def disparity_to_depth(disparity, Q):
    """
    Convert disparity to depth using reprojection matrix Q
    """
    # Reproject to 3D
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    # Extract depth (Z coordinate)
    depth_map = points_3d[:, :, 2]

    # Clip unrealistic values
    depth_map = np.clip(depth_map, 0, 50)  # Assume max depth 50m

    return depth_map
```

### Results

![Depth Estimation](/cv-projects/depth-map.png)
_Left: Input image | Right: Estimated depth map (blue=near, red=far)_

**Performance**:

- Depth accuracy: ±5% error at distances <10m
- Processing speed: 15 FPS on CPU
- Successfully handles:
  - Textureless regions (smoothness constraint in SGBM)
  - Occlusions (left-right consistency check)
  - Reflective surfaces (SAD matching robust to brightness changes)

---

## Overall Takeaways

### Technical Skills Developed

**Core Computer Vision**:

- Feature detection and matching
- Geometric transformations (homography, fundamental/essential matrices)
- 3D reconstruction from 2D images
- Probabilistic modeling (GMMs, Bayesian inference)

**Mathematical Foundations**:

- Linear algebra (SVD, eigendecomposition)
- Optimization (RANSAC, least squares)
- Projective geometry
- Statistics (covariance, PDF estimation)

**Software Engineering**:

- NumPy for efficient array operations
- OpenCV integration
- Algorithm implementation from research papers
- Performance optimization

### Key Insights

**Real Data is Messy**: Theoretical algorithms require extensive tuning and error handling for real images.

**Robustness Matters**: RANSAC, outlier rejection, and regularization are essential for reliable results.

**Geometry Underpins Vision**: Understanding epipolar geometry, projective transformations, and camera models is critical.

**Trade-offs Everywhere**: Speed vs. accuracy, simplicity vs. robustness—every design decision involves trade-offs.

---

## Applications

These techniques are foundational for:

- **Autonomous vehicles**: Depth estimation for obstacle avoidance
- **AR/VR**: 3D reconstruction for virtual object placement
- **Robotics**: Visual odometry and SLAM
- **Photography**: Panorama stitching, depth-of-field effects
- **Medical imaging**: 3D reconstruction from CT/MRI scans

---

## Future Directions

To extend this work:

1. **Deep learning**: Replace handcrafted features with learned representations (CNNs)
2. **Real-time optimization**: GPU acceleration for video-rate processing
3. **Multi-view reconstruction**: Extend to N cameras (full SfM pipeline)
4. **Semantic segmentation**: Combine depth with object recognition

---

## Technologies Used

**Languages**: Python 3.8  
**Libraries**: OpenCV 4.5, NumPy, SciPy, Matplotlib  
**Development**: Jupyter Notebooks, Google Colab  
**Version Control**: Git, GitHub  
**Visualization**: Matplotlib (2D), Open3D (3D point clouds)

---

_These projects demonstrate mastery of fundamental computer vision techniques—essential building blocks for advanced perception systems in robotics and autonomous vehicles._
