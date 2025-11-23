---
title: 'Qubo - Autonomous Underwater Vehicle'
description: 'Competition AUV for RoboSub with ML-enhanced stereo vision system'
image: 'qubo/qubo.jpg'
preview: 'qubo/deep-pool-testing.mp4'
media: 'https://www.youtube.com/embed/srvkYn-s5TI?si=PAD8qaLIVJfSjBxT&amp;start=4'
priority: 10
tags:
  - ROS
  - Computer Vision
  - Machine Learning
  - YOLO
  - Python
  - Team Project
---

## Meet Qubo!

![Qubo at Competition](/qubo/qubo.jpg)
_Qubo: University of Maryland's competition AUV_

Qubo is Robotics @ Maryland's flagship autonomous underwater vehicle, purpose-built for the demands of international RoboSub competition. Representing years of iterative design and engineering across mechanical, electrical, and software disciplines, Qubo embodies the team's commitment to pushing the boundaries of underwater autonomy.

## Competition Overview

**[RoboSub](https://robonation.org/programs/robosub/)** is an international autonomous underwater vehicle competition where teams design, build, and program robots to complete underwater tasks without human intervention. As a member of **Robotics @ Maryland** from 2022-2024, I served as **Perception Lead** on the software team, developing and overseeing the computer vision pipeline that enables our UAV, Qubo, to detect, classify, and localize competition elements.

In the **2024 competition** held in San Diego, California, our team placed **11th out of 40+ teams** from around the world. This is a significant achievement showcasing the effectiveness of our integrated perception, navigation, and control systems.

<video
width="100%"
autoplay
loop
muted
className="rounded-lg shadow-lg"

>

  <source src="/qubo/competition-run.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>
_Qubo operating in the competition pool during RoboSub 2024_

---

<!-- split -->

## Mission Challenges

The competition course simulates real-world underwater scenarios with tasks including:

1. **Gate Navigation**: Pass through a colored gate at varying depths
2. **Buoy Interaction**: Identify and bump specific colored buoys in sequence
3. **Torpedo Firing**: Align with targets and fire projectiles through openings
4. **Bin Dropping**: Drop markers into bins identified by color and shape
5. **Path Following**: Follow a colored line on the pool floor
6. **Octagon Surfacing**: Surface within an octagonal boundary

Each task requires **real-time computer vision, sensor fusion localization, and autonomous behavior execution**—all operating 10-15 feet underwater with no human intervention.

---

## Meet Qubo!

![Qubo at Competition](/qubo/qubo.jpg)
_Qubo: University of Maryland's competition AUV_

**Design Philosophy**: The robot's architecture prioritizes modularity, reliability, and maintainability. A custom aluminum chassis houses waterproof electronics enclosures, pneumatic actuation systems, and an 8-thruster vectored propulsion system capable of full 6-degree-of-freedom motion. Every subsystem, from the custom backplane electrical architecture to the triple-camera perception suite, is designed for rapid iteration and robust performance in the challenging underwater environment.

**Key Capabilities**:

- **Autonomous Navigation**: Multi-sensor fusion (IMU, DVL, pressure, vision) enables precise 3D positioning without GPS
- **Advanced Perception**: Triple-camera stereo vision with ML-enhanced object detection
- **Versatile Manipulation**: Pneumatic-actuated claw, dropper, and torpedo systems for diverse tasks
- **Real-Time Processing**: NVIDIA Jetson Xavier NX running full ROS autonomy stack onboard

**Competition Heritage**: Qubo represents the culmination of the team's RoboSub program, building on lessons learned from previous iterations. The 2024 competition configuration showcased significant advances in perception reliability and autonomous decision-making, contributing to our 11th place finish among 40+ international teams.

![Qubo CAD Model](/qubo/qubo-cad.png)
_Complete CAD model showing mechanical systems integration_

The CAD model shows Qubo's sophisticated mechanical design. The dual side plates form the structural backbone, connecting thruster mounts, camera housings, and the central electronics hull. The pneumatics enclosure sits prominently at the robot's center, distributing compressed air to actuators throughout the chassis. Forward-facing stereo cameras are precisely positioned for optimal depth perception, while the downward camera maintains situational awareness of the pool floor. Every component placement balances hydrodynamic efficiency, center-of-mass considerations, and ease of access for maintenance.

<video
width="100%"
autoplay
loop
muted
controls
className="rounded-lg shadow-lg"

>

  <source src="/qubo/prequalifying.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

_Testing Qubo at the Neutral Buoyancy Research Facility at UMD_

---

## My Role: Computer Vision Architecture

As Perception Lead, I was responsible for designing and implementing Qubo's computer vision pipeline which is the system that translates raw camera data into actionable spatial information for autonomous navigation. My work bridged the gap between low-level image processing and high-level autonomy behaviors.

### System Architecture Evolution

The perception system I developed consists of three integrated subsystems:

**1. Multi-Camera Stereo Vision Setup**

Qubo uses **three exploreHD 3.0 Underwater ROV/AUV cameras**:

- **Two forward-facing cameras**: Stereo pair for depth estimation (baseline ~30cm)
- **One downward-facing camera**: Ground-relative positioning and path following

This configuration provides both long-range target detection and close-range manipulation awareness. The stereo setup enables depth map generation, which is critical for estimating 3D poses of competition elements.

**2. Hybrid ML + Classical CV Pipeline**

Rather than relying solely on traditional color-based detection (which proved unreliable underwater), I architected a hybrid approach:

- **YOLO object detection**: Fine-tuned models for RoboSub-specific objects (gates, buoys, bins, torpedoes)
- **OpenCV refinement**: Traditional CV methods to extract precise geometric features from YOLO regions
- **Stereo correspondence**: Depth estimation from calibrated camera pairs
- **Pose estimation**: 6-DOF pose calculation using feature locations and depth maps

This hybrid approach leverages the robustness of machine learning while maintaining the precision needed for autonomous manipulation tasks.

**3. ROS Integration & Real-Time Processing**

The perception pipeline runs as a collection of ROS nodes:

- **Camera drivers**: Publish synchronized stereo image pairs at 30 FPS
- **Detection nodes**: Run YOLO inference and publish bounding boxes
- **Feature extraction**: OpenCV processing to identify specific task elements
- **Pose estimation**: Transform 2D detections into 3D positions in robot frame
- **Visualization**: Debug overlays published to RViz for real-time monitoring

<video
width="100%"
autoplay
loop
muted
className="rounded-lg shadow-lg"

>

  <source src="/qubo/object-tracking.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>
_Successful tracking of gates, targets and boueys_

---

## Technical Deep Dive: Perception Pipeline

### Challenge: Underwater Vision Complexity

Underwater computer vision presents unique challenges that don't exist in terrestrial robotics:

**Light attenuation**: Water absorbs light wavelengths differently—reds disappear first, shifting everything blue-green at depth.

**Backscatter**: Suspended particles reflect light back to the camera, creating noise similar to driving in fog.

**Refraction**: Camera calibration must account for water's refractive index (1.33), which affects focal length and distortion parameters.

**Dynamic lighting**: Surface ripples create moving caustics and varying illumination.

### Solution: Adaptive Multi-Stage Pipeline

#### Stage 1: YOLO-Based Object Detection

I trained and fine-tuned YOLOv5 models specifically for RoboSub objects:

```python
class PerceptionNode:
    def __init__(self):
        # Load fine-tuned YOLO model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='models/robosub_yolo.pt')
        self.model.conf = 0.6  # Detection confidence threshold
        self.model.iou = 0.45  # NMS IOU threshold

    def detect_objects(self, image):
        """
        Run YOLO inference on image
        Returns: List of Detection objects with class, bbox, confidence
        """
        results = self.model(image)
        detections = []

        for *xyxy, conf, cls in results.xyxy[0]:
            detection = Detection(
                class_id=int(cls),
                class_name=self.model.names[int(cls)],
                bbox=xyxy,
                confidence=float(conf)
            )
            detections.append(detection)

        return detections
```

**Training Data**: I collected and annotated over 2,000 underwater images from previous competitions, pool testing, and synthetic Gazebo renders to create a robust training dataset.

**Performance**: The YOLO model achieves 92% mAP@0.5 on our validation set, with real-time inference at 35 FPS on Jetson Xavier NX.

#### Stage 2: Feature Extraction with OpenCV

YOLO provides coarse bounding boxes, but many tasks require precise geometric features. For example, the gate task requires identifying the exact positions of the gate's vertical posts.

```python
def extract_gate_features(self, image, gate_bbox):
    """
    Extract precise gate post positions from YOLO detection
    """
    # Crop to gate region
    x1, y1, x2, y2 = gate_bbox
    roi = image[y1:y2, x1:x2]

    # Convert to HSV for color-based segmentation
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Adaptive thresholding based on depth
    depth = self.get_current_depth()
    lower_bound, upper_bound = self.adaptive_hsv_range(depth, "orange")

    # Create binary mask
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Morphological operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours representing gate posts
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Filter and identify left/right posts
    posts = self.identify_gate_posts(contours)

    return posts
```

**Key Innovation**: The adaptive HSV thresholding adjusts color ranges based on depth sensor readings, compensating for wavelength-dependent light attenuation. At 3 meters depth, the "orange" range shifts significantly toward brown/yellow compared to surface values.

#### Stage 3: Stereo Depth Estimation

With two forward-facing cameras, I implemented stereo correspondence to generate depth maps:

```python
class StereoDepthEstimator:
    def __init__(self):
        # Load camera calibration from checkerboard calibration
        self.left_camera_matrix = np.load('calib/left_K.npy')
        self.right_camera_matrix = np.load('calib/right_K.npy')
        self.stereo_R = np.load('calib/stereo_R.npy')
        self.stereo_T = np.load('calib/stereo_T.npy')

        # Compute rectification transforms
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.left_camera_matrix, self.left_dist,
            self.right_camera_matrix, self.right_dist,
            (1920, 1080), self.stereo_R, self.stereo_T
        )

        self.Q = Q  # Disparity-to-depth matrix

        # Semi-global block matching for disparity
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    def compute_depth_map(self, left_image, right_image):
        """
        Generate dense depth map from stereo pair
        """
        # Compute disparity
        disparity = self.stereo.compute(left_image, right_image)

        # Convert to depth
        depth_map = cv2.reprojectImageTo3D(disparity, self.Q)

        return depth_map
```

**Camera Calibration**: I performed extensive calibration sessions using checkerboard patterns both in air and underwater to account for refractive distortion. The underwater calibration adjusts the effective focal length by a factor of 1.33.

**Depth Map Quality**: Our stereo setup achieves ±5cm depth accuracy at 2-3 meters range, sufficient for navigation and manipulation planning.

#### Stage 4: 3D Pose Estimation

The final stage converts 2D image features and depth information into 6-DOF poses (position + orientation) in the robot's frame:

```python
def estimate_gate_pose(self, left_posts, right_posts, depth_map):
    """
    Estimate 6-DOF pose of gate from post detections
    """
    # Get 3D positions of gate posts
    left_post_3d = self.project_to_3d(left_posts, depth_map)
    right_post_3d = self.project_to_3d(right_posts, depth_map)

    # Gate center is midpoint between posts
    gate_center = (left_post_3d + right_post_3d) / 2

    # Gate orientation: normal to the plane defined by posts
    gate_width_vector = right_post_3d - left_post_3d
    gate_normal = np.cross(gate_width_vector, np.array([0, 0, 1]))
    gate_normal = gate_normal / np.linalg.norm(gate_normal)

    # Construct pose
    pose = Pose()
    pose.position = gate_center
    pose.orientation = self.vector_to_quaternion(gate_normal)

    # Transform to robot base frame
    pose_robot_frame = self.camera_to_robot_transform @ pose

    return pose_robot_frame
```

This pose information is published on ROS topics where the autonomy system subscribes to plan approach trajectories.

---

## Integration with Autonomy & Control

My perception pipeline doesn't operate in isolation, but rather it's tightly integrated with the autonomy and navigation systems.

### Behavior Tree Integration

The autonomy team uses **BehaviorTree.CPP** to orchestrate high-level task strategies. My perception nodes publish target poses that the behavior trees consume:

**Example: Gate Passage Behavior**

1. Behavior tree checks if gate is visible: `GateDetected` condition node
2. If detected, query gate pose: `GetGatePose` action node (calls my perception service)
3. Navigate to approach waypoint: `MoveToPose` action using navigation system
4. Align with gate: `AlignWithGate` action (continuous pose feedback from perception)
5. Execute passage: `MoveForward` action

The behavior tree makes real-time decisions based on perception data, handling cases like lost detections or ambiguous observations.

### Sensor Fusion with Navigation

While I focused on vision, the navigation team integrates visual data with other sensors:

- **VectorNav VN-100 IMU**: Orientation and angular velocity
- **WaterLinked DVL-A50**: Velocity measurements (Doppler effect)
- **Bar02 Pressure Sensor**: Depth estimation

The Extended Kalman Filter (EKF) fuses these modalities to track Qubo's full 6-DOF state, which is then used to transform vision-based target poses into a global reference frame for path planning.

---

## Simulation-Driven Development

Before deploying to hardware, I extensively tested the perception pipeline in **Gazebo**, a physics-based simulator.

### Custom Gazebo Environment

I utilized a team built custom simulation of the RoboSub competition pool:

**Environment Features**:

- Particle based water physics simulation
- Task props (gates, buoys, bins) at competition scale
- Camera sensor plugins with configurable noise and distortion

**Simulation vs. Reality Gap**:

One critical aspect of sim-to-real transfer is ensuring that YOLO models trained on synthetic data generalize to real underwater imagery. I addressed this through:

1. **Domain randomization**: Vary lighting, water turbidity, and prop textures in simulation
2. **Synthetic + real training mix**: 60% real images, 40% synthetic images in training set
3. **Progressive deployment**: Test on recorded pool videos before live testing

<video
width="100%"
autoplay
loop
muted
className="rounded-lg shadow-lg"

>

  <source src="/qubo/qubo-sim.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>

_Gazebo simulation environment with task elements_

### Benefits of Simulation

- **Rapid iteration**: Test perception changes in minutes vs. hours of pool time
- **Reproducible scenarios**: Exact same conditions for comparative testing
- **Edge case testing**: Simulate rare scenarios (e.g., partial occlusions, extreme lighting)
- **Algorithm validation**: Verify YOLO inference, stereo matching, and pose estimation logic

**Performance**: Perception algorithms developed in simulation achieved **>85% transfer success** when deployed to real hardware. I found that the detections that worked in Gazebo reliably worked in the pool.

---

## Pool Testing & Validation

Weekly pool testing at the university's Eppley Recreation Center was critical for validating the perception system under real conditions.

### Testing Protocol

**Pre-dive checklist**:

1. Verify camera synchronization (stereo requires <10ms timing offset)
2. Confirm YOLO model loaded correctly on Jetson
3. Check ROS topic publication rates (perception nodes must maintain 30 Hz)
4. Calibrate pressure sensor zero-point

**During testing**:

- Run autonomous missions with incremental task complexity
- Record all sensor data (rosbags with camera feeds, detections, poses)
- Monitor real-time RViz visualization for detection failures

**Post-dive analysis**:

- Replay rosbags to identify missed detections
- Measure detection latency and pose estimation accuracy
- Iterate on model thresholds, OpenCV parameters, or YOLO retraining

<video
Width="70%"
autoplay
loop
muted
className="rounded-lg shadow-lg"

>

  <source src="/qubo/pool-testing.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>
_In-pool testing session at Eppley Recreation Center_

### Key Challenges Solved

**Challenge 1: Stereo Calibration Drift**  
**Problem**: Camera mounts flexed slightly under pressure, invalidating stereo calibration  
**Solution**: Switched to more rigid mounts and added runtime stereo recalibration using visual features on known competition props

**Challenge 2: YOLO Overfitting**  
**Problem**: Initial YOLO model achieved 98% accuracy in simulation but only 65% in the pool  
**Solution**: Expanded training data with more real underwater images and aggressive data augmentation (color jitter, Gaussian blur, synthetic particulate noise)

**Challenge 3: Real-Time Performance on Jetson**  
**Problem**: Full-resolution stereo matching was too slow (8 FPS), causing control lag  
**Solution**: Implemented region-of-interest (ROI) stereo matching. This meant that we are only computing the depth for regions where YOLO detected objects, reducing computational load by 70%

---

## Competition Performance

At **RoboSub 2024** in San Diego, our integrated system performed well:

**Successful gate passage** on first attempt (stereo depth + YOLO gate detection)  
**Buoy identification and bumping** (3/4 targets, missed one due to lighting)  
**Accurate path following** using downward camera + OpenCV line detection  
**11th place finish** semifinalist out of 40+ teams

**Perception-Specific Metrics**:

- Gate detection: 100% success rate (5/5 runs)
- Buoy detection: 88% success rate (14/16 attempts)
- Average detection latency: 45ms (camera to pose estimate)
- False positive rate: <2% (very few spurious detections)

**Areas for Improvement**:

- Bin classification accuracy (struggled with similar colors at depth)
- Long-range target detection (YOLO confidence dropped below threshold >5m away)
- Robustness to water turbidity (thruster wash degraded depth map quality)

![Competition Results](/qubo/robosub-team-pic.png)
_Team celebrating semifinalist finish at RoboSub 2024_

---

## Technical Specifications

### Hardware

- **Frame**: Custom aluminum chassis (waterproof to 20m)
- **Propulsion**: 8× Blue Robotics T200 thrusters (vectored thrust configuration)
- **Cameras**: 3× exploreHD 3.0 Underwater ROV/AUV cameras
- **Compute**: NVIDIA Jetson Xavier NX (6-core CPU + 384-core GPU)
- **Sensors**:
  - VectorNav VN-100 IMU (orientation)
  - WaterLinked DVL-A50 (velocity)
  - Bar02 Pressure Sensor (depth)
- **Power**: 16V LiPo battery (60 min runtime)

### Software Stack

- **Framework**: ROS Noetic (modular node architecture)
- **Computer Vision**:
  - OpenCV 4.5 (classical CV algorithms)
  - PyTorch 1.10 + YOLOv5 (deep learning)
  - Camera calibration tools (stereo calibration)
- **Simulation**: Gazebo 11 (physics-based testing)
- **Languages**: Python 3.8 (perception), C++ (controls)
- **Autonomy**: BehaviorTree.CPP (decision-making)

---

## Lessons Learned

### Technical Insights

**Hybrid approaches win**: Pure ML or pure classical CV both have limitations underwater. YOLO provides robust detection, but OpenCV refinement gives the precision needed for manipulation.

**Calibration is critical**: Underwater stereo vision requires meticulous calibration. Even small errors in camera parameters cause significant depth estimation errors at range.

**Test in realistic conditions**: Simulation is invaluable, but nothing replaces real pool testing. Turbidity, lighting dynamics, and pressure effects can't be fully simulated.

### Team Collaboration

As Perception Lead, I worked closely with:

- **Navigation team**: Defined pose message formats and transform frames
- **Autonomy team**: Designed perception service APIs for behavior trees
- **Electrical team**: Specified camera mounting requirements and synchronization
- **Mechanical team**: Iterated on camera housing and thruster placement to minimize vibration

**Documentation**: I created comprehensive technical documentation for future perception leads, including model training guides, calibration procedures, and debugging workflows.

### Personal Growth

- **Systems thinking**: Learned to design perception not in isolation, but as part of a larger robotics architecture
- **Real-time constraints**: Balanced perception accuracy with computational performance (latency matters for control)
- **Leadership**: Coordinated perception efforts across team members, assigned tasks, and mentored newer members

---

## Impact & Future Directions

The perception system I developed serves as the foundation for future Qubo iterations. The 2025 team is building upon this work:

- **Transformer-based models**: Replacing YOLO with more advanced architectures (DINO, DETR)
- **Visual SLAM**: Implementing simultaneous localization and mapping for better long-term navigation
- **Multi-object tracking**: Maintaining temporal consistency across detections (Kalman filtering on target tracks)
- **Active vision**: Pan-tilt camera mounts for expanding field of view

---

## Links

📄 **[RoboSub Competition](https://robonation.org/programs/robosub/)**: Official competition website  
🌐 **[Robotics @ Maryland](https://robotics.umd.edu/)**: Team website with full technical documentation

![Team Photo](/qubo/team-photo.jpg)
_Robotics @ Maryland 2024 Team_

---

## Technologies & Skills

**Computer Vision**: YOLO object detection, OpenCV, Stereo vision, Camera calibration, Feature extraction  
**Machine Learning**: PyTorch, YOLOv5 fine-tuning, Dataset curation, Model optimization  
**Robotics**: ROS, Sensor fusion, Coordinate transforms (tf), Pose estimation  
**Simulation**: Gazebo, RViz, URDF modeling  
**Programming**: Python, C++, Bash scripting  
**Tools**: Git, Docker, CUDA optimization, Jupyter notebooks

---

_Perception Software Lead, RoboSub 2024 Semi-finalist_
