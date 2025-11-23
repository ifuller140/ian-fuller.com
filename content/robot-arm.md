---
title: 'Robot Kinematics & Manipulation'
description: '6-DOF robotic arm pick-and-place with computer vision'
image: 'robot-arm/robot.jpg'
preview: 'robot-arm/preview.mp4'
media: 'https://www.youtube.com/embed/yOQUNYTjXFI'
priority: 6
tags:
  - UR3e Robot
  - ROS
  - Python
  - OpenCV
  - Kinematics
links:
  - text: View on GitHub
    href: https://github.com/ENME480/Lab-Code/blob/main/Final%20Project/Final_Project_Details.md
---

## Project Overview

Industrial robots operate in cartesian space (X, Y, Z positions) but are controlled in joint space (6 motor angles). This project required me to implement the mathematical transformations that bridge these two worlds, **forward and inverse kinematics**, for a **Universal Robots UR3e** collaborative robot arm.

I designed and programmed a complete pick-and-place system that:

1. Uses a **stationary camera** to detect colored blocks
2. Computes **coordinate transformations** from camera to robot frame
3. Plans **collision-free trajectories** to grasp blocks
4. **Stacks blocks** by color in designated locations

This demonstrates fundamental robotics skills: perception, planning, and control integration on professional industrial hardware.

![UR3e Setup](/robot-arm/robot-setup.jpg)
_UR3e collaborative robot with overhead camera system_

---

<!-- split -->

## System Architecture

### Hardware Components

**UR3e Collaborative Robot**:

- **6 degrees of freedom** (6 revolute joints)
- **3 kg payload** capacity
- **500 mm reach**
- **±0.1 mm repeatability**

**Perception System**:

- **Intel RealSense D435** depth camera (mounted overhead)
- **1920×1080 resolution** RGB stream
- **87° × 58° field of view**

**End Effector**:

- **Robotiq 2F-85** parallel gripper
- **85 mm stroke**
- **Force sensing** for gentle grasping

![System Diagram](/robot-arm/system-diagram.png)
_Data flow from camera detection to robot motion_

---

## Mathematical Foundation: Kinematics

### Forward Kinematics

**Problem**: Given 6 joint angles (θ₁, θ₂, ..., θ₆), where is the end effector in cartesian space?

**Solution**: Denavit-Hartenberg (DH) parameters + homogeneous transformations

I derived the DH parameter table for the UR3e:

```
┌───────┬────────┬───────────┬───────────┬───────────┐
│ Joint │   θ    │     d     │     a     │     α     │
├───────┼────────┼───────────┼───────────┼───────────┤
│   1   │   θ₁   │  0.15185  │     0     │    π/2    │
│   2   │   θ₂   │     0     │ -0.24355  │     0     │
│   3   │   θ₃   │     0     │ -0.21325  │     0     │
│   4   │   θ₄   │  0.13105  │     0     │    π/2    │
│   5   │   θ₅   │  0.08535  │     0     │   -π/2    │
│   6   │   θ₆   │  0.09465  │     0     │     0     │
└───────┴────────┴───────────┴───────────┴───────────┘
```

**Transformation Matrix Construction**:

```python
def dh_transform(theta, d, a, alpha):
    """
    Create 4x4 transformation matrix from DH parameters
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),                 np.cos(alpha),                d              ],
        [0,              0,                             0,                            1              ]
    ])

def forward_kinematics(joint_angles):
    """
    Compute end effector pose from joint angles
    """
    # DH parameters for UR3e
    dh_params = [
        [joint_angles[0], 0.15185,  0,        np.pi/2],
        [joint_angles[1], 0,       -0.24355,  0      ],
        [joint_angles[2], 0,       -0.21325,  0      ],
        [joint_angles[3], 0.13105,  0,        np.pi/2],
        [joint_angles[4], 0.08535,  0,       -np.pi/2],
        [joint_angles[5], 0.09465,  0,        0      ]
    ]

    # Multiply transformation matrices
    T = np.eye(4)
    for params in dh_params:
        T = T @ dh_transform(*params)

    return T
```

**Result**: The final transformation matrix `T` contains:

- **Position**: T[0:3, 3] gives [X, Y, Z]
- **Orientation**: T[0:3, 0:3] gives rotation matrix

![Forward Kinematics Visualization](/robot-arm/fk-visualization.png)
_Coordinate frames for each joint (visualized in RViz)_

### Inverse Kinematics (IK)

**Problem**: Given a desired end effector pose, what joint angles achieve it?

**Challenge**: This is **much harder** than forward kinematics:

- **Non-linear equations** (no closed-form solution)
- **Multiple solutions** exist (robot can reach same point with different configurations)
- **Joint limits** and **singularities** must be avoided

**My Approach**: Numerical inverse kinematics using Newton-Raphson iteration

```python
def inverse_kinematics(target_pose, initial_guess):
    """
    Compute joint angles for desired end effector pose
    Uses iterative numerical solver
    """
    joint_angles = np.array(initial_guess)
    max_iterations = 100
    tolerance = 1e-4

    for iteration in range(max_iterations):
        # Compute current end effector pose
        current_pose = forward_kinematics(joint_angles)

        # Compute pose error
        position_error = target_pose[0:3, 3] - current_pose[0:3, 3]
        orientation_error = rotation_error(target_pose[0:3, 0:3], current_pose[0:3, 0:3])
        error = np.concatenate([position_error, orientation_error])

        # Check convergence
        if np.linalg.norm(error) < tolerance:
            return joint_angles

        # Compute Jacobian matrix (relates joint velocities to end effector velocities)
        J = compute_jacobian(joint_angles)

        # Newton-Raphson update: Δθ = J⁺ · error (J⁺ is pseudoinverse)
        delta_theta = np.linalg.pinv(J) @ error
        joint_angles += delta_theta

        # Enforce joint limits
        joint_angles = np.clip(joint_angles, joint_limits_low, joint_limits_high)

    raise Exception("IK did not converge")
```

**Jacobian Computation**: The Jacobian matrix relates joint velocities to end effector velocities. I computed it using **numerical differentiation**:

```python
def compute_jacobian(joint_angles):
    """
    6x6 Jacobian matrix: dX/dθ
    """
    J = np.zeros((6, 6))
    delta = 1e-6

    for i in range(6):
        # Perturb joint i slightly
        theta_plus = joint_angles.copy()
        theta_plus[i] += delta

        theta_minus = joint_angles.copy()
        theta_minus[i] -= delta

        # Finite difference approximation
        pose_plus = forward_kinematics(theta_plus)
        pose_minus = forward_kinematics(theta_minus)

        J[:, i] = (pose_plus_to_vector(pose_plus) - pose_plus_to_vector(pose_minus)) / (2 * delta)

    return J
```

---

## Camera-to-Robot Calibration

The camera sees blocks in **camera coordinates**. The robot needs positions in **robot base coordinates**. I computed the transformation between these frames using **homography**.

### Calibration Process

1. **Place calibration markers** at known robot base positions
2. **Detect markers** in camera image
3. **Compute homography matrix** H that maps camera pixels to robot coordinates

```python
def compute_homography(camera_points, robot_points):
    """
    Compute 3x3 homography matrix from point correspondences
    """
    # Use OpenCV's findHomography (implements DLT algorithm)
    H, status = cv2.findHomography(camera_points, robot_points, cv2.RANSAC)
    return H

def camera_to_robot(pixel_coords, homography):
    """
    Transform pixel coordinates to robot base frame
    """
    # Convert to homogeneous coordinates
    pixel_homogeneous = np.array([pixel_coords[0], pixel_coords[1], 1])

    # Apply homography
    robot_homogeneous = homography @ pixel_homogeneous

    # Convert back to cartesian
    robot_coords = robot_homogeneous[:2] / robot_homogeneous[2]

    return robot_coords
```

![Calibration Setup](/robot-arm/calibration.png)
_Calibration markers visible in both camera and robot frames_

**Validation**: I verified calibration accuracy by:

- Commanding robot to touch calibration points
- Measuring error between actual and commanded positions
- **Result**: <2mm average error across workspace

---

## Computer Vision: Block Detection

The camera must detect colored blocks and determine their positions and orientations.

### Color Segmentation

I used **HSV color space** (more robust to lighting changes than RGB):

```python
def detect_blocks(image):
    """
    Detect colored blocks in image
    Returns: List of (color, center_x, center_y, orientation)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    blocks = []
    for color_name, (lower_hsv, upper_hsv) in color_ranges.items():
        # Create binary mask for this color
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Too small
                continue

            # Compute center
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            # Compute orientation (fit minimum area rectangle)
            rect = cv2.minAreaRect(contour)
            angle = rect[2]

            blocks.append({
                'color': color_name,
                'pixel_x': cx,
                'pixel_y': cy,
                'orientation': angle
            })

    return blocks
```

![Block Detection](/robot-arm/block-detection.png)
_Camera view with detected blocks highlighted_

**Color Ranges** (tuned experimentally):

```python
color_ranges = {
    'red':    (np.array([0, 100, 100]),   np.array([10, 255, 255])),
    'blue':   (np.array([100, 100, 100]), np.array([130, 255, 255])),
    'green':  (np.array([40, 100, 100]),  np.array([80, 255, 255])),
    'yellow': (np.array([20, 100, 100]),  np.array([40, 255, 255]))
}
```

---

## Motion Planning & Execution

### Pick-and-Place Sequence

For each block:

1. **Approach**: Move to position above block (Z = 0.3m)
2. **Descend**: Lower to grasp height (Z = 0.05m)
3. **Grasp**: Close gripper
4. **Lift**: Raise to safe height (Z = 0.3m)
5. **Transport**: Move to stacking location
6. **Place**: Lower and release
7. **Retract**: Return to home position

```python
def pick_and_place(block_info, stack_location):
    """
    Execute pick-and-place motion for a block
    """
    # Convert block position from camera to robot frame
    robot_x, robot_y = camera_to_robot(
        (block_info['pixel_x'], block_info['pixel_y']),
        homography_matrix
    )

    # Define waypoints
    approach_pose = [robot_x, robot_y, 0.3, 0, np.pi, 0]  # Above block
    grasp_pose = [robot_x, robot_y, 0.05, 0, np.pi, 0]    # At block height
    lift_pose = [robot_x, robot_y, 0.3, 0, np.pi, 0]      # Lifted
    place_pose = [stack_location[0], stack_location[1], 0.3, 0, np.pi, 0]

    # Execute motion sequence
    move_to_pose(approach_pose)
    move_to_pose(grasp_pose)
    close_gripper()
    move_to_pose(lift_pose)
    move_to_pose(place_pose)
    open_gripper()
    move_home()
```

### Trajectory Smoothing

Direct point-to-point motion causes jerky movements. I implemented **joint space interpolation** for smooth trajectories:

```python
def move_to_pose(target_pose, duration=3.0):
    """
    Smoothly move to target pose over specified duration
    """
    # Solve IK for target
    target_joints = inverse_kinematics(target_pose, current_joint_angles)

    # Generate trajectory (quintic polynomial for smooth acc/decel)
    trajectory = generate_quintic_trajectory(
        start_joints=current_joint_angles,
        end_joints=target_joints,
        duration=duration,
        timestep=0.01
    )

    # Execute trajectory
    for waypoint in trajectory:
        robot.movej(waypoint)
        time.sleep(0.01)
```

---

## Results & Performance

### Success Metrics

Tested over 50 pick-and-place cycles:

- **98% grasp success rate** (1 dropped block due to gripper slip)
- **100% color sorting accuracy** (no misclassification)
- **±3mm placement accuracy** (measured with calipers)

### Speed Optimization

Initial implementation: **45 seconds per block**  
After optimization: **22 seconds per block**

**Optimizations**:

- Reduced Z-axis safety margin (0.3m → 0.2m)
- Increased motion speed (50% → 80% of max)
- Parallelized gripper actuation with motion planning

![Stacking Result](/robot-arm/stacked-blocks.jpg)
_Successfully stacked blocks sorted by color_

---

## Challenges & Solutions

### Challenge 1: IK Convergence Failures

**Problem**: Inverse kinematics failed to converge for ~10% of target poses

**Root cause**: Poor initial guess led solver into local minimum

**Solution**:

- Used multiple initial guesses (4 different elbow configurations)
- Selected solution with lowest joint motion from current pose
- Added fallback to Jacobian transpose method if Newton-Raphson failed

### Challenge 2: Lighting Sensitivity

**Problem**: Block detection failed under bright overhead lights (color washout)

**Solution**:

- Added polarizing filter to camera lens
- Adjusted exposure to prevent saturation
- Implemented adaptive thresholding based on ambient light measurement

### Challenge 3: Gripper Force Control

**Problem**: Gripper either crushed blocks (too much force) or dropped them (too little)

**Solution**:

- Used gripper's force feedback to detect contact
- Implemented "grasp-until-force" strategy (close until 20N detected)
- Added compliance in gripper jaws (3D printed TPU pads)

---

## Key Learnings

**Mathematical Foundations Matter**: Understanding the theory behind DH parameters and Jacobians was essential for debugging when things went wrong.

**Sensor Calibration is Critical**: Spent 20% of project time on camera calibration—it paid off with reliable detection.

**Industrial Robots are Precise**: The UR3e's repeatability meant once I got the math right, it worked consistently.

**Simulation Helps**: Tested IK algorithms in ROS simulation before touching hardware, catching bugs early.

---

## Technologies Used

**Hardware**: Universal Robots UR3e, Robotiq 2F-85 Gripper, Intel RealSense D435  
**Software**: ROS (Noetic), Python 3.8, OpenCV 4.5, NumPy  
**Math Libraries**: SciPy (optimization), SymPy (symbolic kinematics verification)  
**Visualization**: RViz, Matplotlib  
**Development**: VS Code, Git

---

## Video Demonstration

📹 **[Watch Full Demo](https://www.youtube.com/embed/yOQUNYTjXFI)**: See the complete pick-and-place operation

---

_This project demonstrates practical application of robotics fundamentals: kinematics, perception, and control—skills directly transferable to industrial automation and manipulation research._
