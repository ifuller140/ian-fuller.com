---
title: 'Autonomous Fire Truck'
description: 'Fully autonomous robot for candle extinguishing and terrain navigation'
image: 'fire-truck/fire-truck.jpg'
preview: 'fire-truck/preview.mp4'
priority: 9
media: 'https://www.youtube.com/embed/XeLIS-Nzu3Y'
tags:
  - Arduino
  - SolidWorks
  - 3D Printing
  - Autonomous Systems
  - C++
---

## Project Overview

For my introduction to engineering course, my team was challenged to design, build, and program a fully **autonomous robot** capable of navigating difficult terrain, detecting fires, determining topography, and extinguishing specific candles—all without human intervention after the start signal.

As **mechanical lead and primary programmer**, I designed the robot's unique **folding platform extinguishing system**, integrated all sensors, and implemented the **state-machine control logic** in Arduino C++. The result was a robot that successfully completed all mission objectives in under 2 minutes.

![Fire Truck Final Design](/fire-truck/fire-truck.jpg)
_Final autonomous fire truck with extended suppression platform_

---

<!-- split -->

## Mission Objectives

The robot must autonomously:

1. **Topography Sensing**: Determine the orientation of a block with candles at varying heights (sides A, B, or C)
2. **Fire Sensing**: Count the number of lit candles on the block
3. **Fire Suppression**: Extinguish all candles **except** the center candle
4. **Localization & Navigation**: Use an overhead camera system to navigate to target locations on a 10'×10' arena

**Scoring**: Points awarded for accuracy and speed. Incorrect topography detection = disqualification.

![Mission Arena](/fire-truck/arena-overview.png)
_Competition arena layout with obstacles and target zones_

---

## Design Philosophy

### Mechanical Innovation: Folding Platform System

Traditional approaches used fixed water nozzles or fans. I designed a **mechanically actuated suppression platform** that:

- **Folds over the robot** during navigation (compact profile)
- **Extends over the candle block** during suppression (36" reach)
- **Conforms to varying candle heights** using suspended nodes
- **Preserves the center candle** via a strategically placed hole

![Platform Mechanism](/fire-truck/preview.gif)
_Animation showing platform extension and retraction_

**Why This Works**:

- **No consumables**: No water tanks or CO₂ cartridges to refill
- **Reliability**: Purely mechanical—fewer points of failure than pneumatics
- **Speed**: 3-second deployment time (faster than aiming nozzles)

### Electronic Architecture

The robot uses an **Arduino Uno** as the central controller, coordinating:

**Sensors**:

- 2× HC-SR04 ultrasonic sensors (topography detection)
- 4× DHT20 temperature/humidity sensors (fire detection)
- WiFi module (position data from overhead camera)
- IMU (orientation stabilization during motion)

**Actuators**:

- 2× DC motors with L298N driver (differential drive)
- 1× Servo motor (platform actuation)
- Kill switches (safety requirement)

![Electronics Diagram](/fire-truck/electrical-schematic.png)
_Complete electrical schematic showing all connections_

**Power Management**:

- 12V LiPo battery (split circuit for motors and logic)
- Step-down converter (12V → 5V for Arduino)
- Separate kill switches for drive motors and logic board

---

## Technical Implementation

### 1. Topography Sensing

The candle block has three possible orientations, distinguished by height differences:

![Topography Variants](/fire-truck/topography.png)
_Three possible block orientations (A, B, C)_

**Sensing Strategy**:

I mounted two ultrasonic sensors at specific heights to measure distances to the **center** and **right** portions of the block.

```cpp
// Topography detection algorithm
char detectTopography() {
    float centerDist = readUltrasonic(CENTER_SENSOR_PIN);
    float rightDist = readUltrasonic(RIGHT_SENSOR_PIN);

    float diff = centerDist - rightDist;

    if (diff < -THRESHOLD) {
        return 'A';  // Center closer than right (high center)
    } else if (diff > THRESHOLD) {
        return 'B';  // Center farther than right (low center)
    } else {
        return 'C';  // Center same as right (flat)
    }
}

float readUltrasonic(int trigPin) {
    // Trigger ultrasonic pulse
    digitalWrite(trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(trigPin, HIGH);
    delayMicroseconds(10);
    digitalWrite(trigPin, LOW);

    // Measure echo time
    long duration = pulseIn(trigPin + 1, HIGH, TIMEOUT);

    // Convert to distance (cm)
    float distance = duration * 0.034 / 2;

    return distance;
}
```

**Validation**: Tested on 50+ measurement cycles with 100% accuracy.

![Ultrasonic Mounting](/fire-truck/ultrasonic-sensors.png)
_Custom 3D printed mounts position sensors at correct heights_

### 2. Fire Sensing

Each of the four suppression nodes contains a **DHT20 temperature sensor**. When positioned over a candle, the sensor detects elevated temperature.

**Detection Algorithm**:

```cpp
int countFires() {
    int fireCount = 0;
    float ambientTemp = readAmbientTemperature();  // Baseline from IMU

    // Check each suppression node
    for (int i = 0; i < 4; i++) {
        float temp = dht[i].readTemperature();

        // If temp > 10°C above ambient, candle detected
        if (temp > ambientTemp + FIRE_THRESHOLD) {
            fireCount++;
        }
    }

    return fireCount;
}
```

**Challenge**: Sensors heat up from proximity to previously extinguished candles (residual heat).

**Solution**:

- Implemented **temporal filtering**: Require elevated temp for 3 consecutive readings
- Added **heat dissipation delay**: Wait 2 seconds between node readings

### 3. Fire Suppression

The suppression platform uses four **aluminum-foil-wrapped nodes** suspended on 3D printed rods with ball joints.

![Suppression Nodes](/fire-truck/suppression-nodes.png)
_Four suppression nodes with flexible suspension_

**How It Works**:

1. Platform extends over block (servo rotates arm)
2. Nodes lower onto candles (gravity + flexible rods)
3. Foil cups smother flames (oxygen deprivation)
4. Center hole ensures middle candle untouched
5. Platform retracts after 5-second dwell time

**Mechanical Design**:

The platform is supported by four **acrylic beams** that rotate on **ball bearings** (8 total). A servo-driven arm pulls two beams via zip-ties, causing the entire platform to fold.

```cpp
void extinguishFires() {
    // Extend platform
    platformServo.write(EXTENDED_ANGLE);  // 90 degrees
    delay(3000);  // Allow platform to fully extend

    // Wait for extinguishment
    delay(5000);  // 5 seconds to smother flames

    // Retract platform
    platformServo.write(RETRACTED_ANGLE);  // 0 degrees
    delay(3000);
}
```

![Platform CAD](/fire-truck/platform-cad.png)
_SolidWorks model showing platform mechanism and servo arm_

### 4. Localization & Navigation

The robot uses an **ArUco marker** on its top surface, visible to an overhead camera system that provides real-time position and orientation.

**Position Data Format** (received via WiFi):

```cpp
struct Position {
    float x;        // X coordinate (meters)
    float y;        // Y coordinate (meters)
    float theta;    // Heading angle (radians)
};
```

**Navigation Controller**:

I implemented a simple **proportional controller** for waypoint following:

```cpp
void navigateToWaypoint(float targetX, float targetY) {
    while (distanceToWaypoint(targetX, targetY) > WAYPOINT_TOLERANCE) {
        // Get current position from camera
        Position current = updatePosition();

        // Calculate desired heading
        float desiredHeading = atan2(targetY - current.y, targetX - current.x);
        float headingError = normalizeAngle(desiredHeading - current.theta);

        // Proportional control
        int leftSpeed = BASE_SPEED - (int)(KP_TURN * headingError);
        int rightSpeed = BASE_SPEED + (int)(KP_TURN * headingError);

        // Constrain speeds
        leftSpeed = constrain(leftSpeed, -MAX_SPEED, MAX_SPEED);
        rightSpeed = constrain(rightSpeed, -MAX_SPEED, MAX_SPEED);

        // Apply to motors
        setMotorSpeeds(leftSpeed, rightSpeed);

        delay(50);  // 20 Hz control loop
    }

    // Stop at waypoint
    setMotorSpeeds(0, 0);
}
```

**Tuning**: Adjusted `KP_TURN` gain through trial-and-error testing to minimize overshoot while maintaining responsiveness.

---

## Software Architecture: State Machine

The robot's behavior is controlled by a **finite state machine** that sequences mission tasks:

```cpp
enum State {
    IDLE,
    NAVIGATE_TO_BLOCK,
    DETECT_TOPOGRAPHY,
    APPROACH_BLOCK,
    SENSE_FIRES,
    EXTINGUISH,
    RETREAT,
    MISSION_COMPLETE
};

State currentState = IDLE;

void loop() {
    switch (currentState) {
        case IDLE:
            // Wait for start signal
            if (startButtonPressed()) {
                currentState = NAVIGATE_TO_BLOCK;
            }
            break;

        case NAVIGATE_TO_BLOCK:
            navigateToWaypoint(BLOCK_X, BLOCK_Y);
            currentState = DETECT_TOPOGRAPHY;
            break;

        case DETECT_TOPOGRAPHY:
            char topo = detectTopography();
            transmitTopography(topo);  // Send to judges
            currentState = APPROACH_BLOCK;
            break;

        case APPROACH_BLOCK:
            // Fine positioning using ultrasonic feedback
            approachUntilDistance(TARGET_DISTANCE);
            currentState = SENSE_FIRES;
            break;

        case SENSE_FIRES:
            int fireCount = countFires();
            transmitFireCount(fireCount);
            currentState = EXTINGUISH;
            break;

        case EXTINGUISH:
            extinguishFires();
            currentState = RETREAT;
            break;

        case RETREAT:
            navigateToWaypoint(HOME_X, HOME_Y);
            currentState = MISSION_COMPLETE;
            break;

        case MISSION_COMPLETE:
            celebratoryBeep();
            // Stop execution
            break;
    }
}
```

![State Machine Diagram](/fire-truck/pipeline.png)
_State transition diagram showing mission flow_

**Benefits of State Machine Architecture**:

- **Clear logic flow**: Easy to debug and modify
- **Modularity**: Each state is self-contained
- **Testability**: Can test individual states in isolation
- **Extensibility**: Adding new behaviors is straightforward

---

## Mechanical Design Details

### Drivetrain

**Configuration**: Differential drive (2 powered wheels + 2 caster wheels)

**Wheel Selection**:

- Diameter: 3.15" (80mm)
- Material: Rubber (high traction on arena floor)
- Rationale: Large enough to traverse obstacles, small enough for tight turns

![Drivetrain Design](/fire-truck/drivetrain.jpg)
_Motor-driven rear wheels with front caster wheels_

**Motor Specifications**:

- Type: 12V DC geared motors
- Torque: 5 kg·cm
- RPM: 200 (provides good balance of speed and torque)
- Control: L298N H-bridge driver

### Platform Mechanism

The folding platform was the most complex mechanical subsystem.

**Design Requirements**:

- Extend 18" beyond robot body
- Support 200g (weight of suppression nodes)
- Deploy in <3 seconds
- Withstand 50+ actuation cycles

**Material Selection**:

- Frame: 3D printed PETG (strength + temperature resistance)
- Platform surface: Laser-cut plywood (lightweight + rigid)
- Beams: Acrylic rod (transparency aids alignment during assembly)
- Joints: Steel ball bearings (low friction, high durability)

![Platform Extended CAD](/fire-truck/full-body-view-extended-cad.png)
_Full CAD model showing extended platform configuration_

**Servo Sizing**:

- Required torque: 2 kg·cm (calculated from platform weight and lever arm)
- Selected: Standard servo with 3 kg·cm (safety margin)

### Suppression Nodes

Each node consists of:

- **Housing**: 3D printed PETG shell
- **Sensor**: DHT20 temperature/humidity sensor
- **Covering**: Aluminum foil (reflects heat, conforms to candle tops)
- **Rod**: 3D printed with ball joint (allows tilting)

![Node Design](/fire-truck/node-cross-section.png)
_Cross-section of suppression node showing sensor placement_

---

## Testing & Iteration

### Prototype Evolution

**Version 1**: Fan-based air blast

- ❌ Insufficient force to extinguish candles
- ❌ Blew out center candle (disqualification)

**Version 2**: Mechanical smothering (final design)

- ✅ Reliable extinguishment
- ✅ No consumables
- ✅ Center candle preserved

### Test Results

**Arena Testing** (15 full mission runs):

- **Topography Detection**: 15/15 correct (100%)
- **Fire Count**: 14/15 correct (93%, 1 false positive)
- **Extinguishment**: 12/15 successful (80%)
- **Navigation**: Average positioning error <4 cm

**Competition Performance**:

- Mission completion time: **1 minute 47 seconds**
- Awarded **Best Mission Award**

![Competition Result](/fire-truck/competition-action.jpg)
_Robot during competition run_

---

## Challenges Overcome

### Challenge 1: Sensor Noise

**Problem**: Ultrasonic sensors gave erratic readings near metal obstacles.

**Solution**:

- Implemented **median filtering** (take 5 readings, use middle value)
- Added **outlier rejection** (discard readings >2σ from mean)

```cpp
float filteredUltrasonic(int pin) {
    const int SAMPLES = 5;
    float readings[SAMPLES];

    // Collect samples
    for (int i = 0; i < SAMPLES; i++) {
        readings[i] = readUltrasonic(pin);
        delay(10);
    }

    // Sort array
    qsort(readings, SAMPLES, sizeof(float), compareFloat);

    // Return median
    return readings[SAMPLES / 2];
}
```

### Challenge 2: Platform Binding

**Problem**: Platform mechanism occasionally jammed during deployment.

**Root Cause**: Acrylic beams slightly warped from heat during assembly.

**Solution**:

- Replaced warped beams with properly annealed acrylic
- Added **anti-backlash spring** to servo arm (maintains tension)
- Increased bearing clearances by 0.2mm

### Challenge 3: WiFi Latency

**Problem**: Position updates from camera had 100-200ms latency.

**Solution**:

- **Predictive positioning**: Estimate position based on commanded motor speeds during latency period
- **Increased control loop frequency**: 50ms → 20ms (compensates for stale data)

---

## What I Learned

### Technical Skills

**Embedded Systems**:

- Arduino programming in C++
- Sensor integration and signal processing
- Motor control with PWM
- Serial communication protocols

**Mechanical Design**:

- SolidWorks CAD modeling
- Mechanism design (linkages, bearings, joints)
- Material selection for mechanical properties
- 3D printing for rapid prototyping

**System Integration**:

- Coordinating multiple subsystems (sensors, actuators, logic)
- Debugging hardware/software interaction issues
- Testing methodologies for robotics systems

### Engineering Process

**Iterative Design**: Initial concept rarely works perfectly—embrace iteration

**Testing is Critical**: Spent 50% of project time on testing (paid off with reliable performance)

**Documentation**: Maintained detailed build log—invaluable during troubleshooting

**Team Collaboration**: Regular check-ins with team prevented integration issues

---

## Future Improvements

If I were to redesign this robot:

1. **Add obstacle detection**: Integrate front-facing distance sensors for collision avoidance
2. **Improve fire sensing**: Use IR flame sensors (faster response than temperature)
3. **Upgrade controller**: Move to Raspberry Pi for onboard vision processing
4. **Enhance navigation**: Implement SLAM for autonomous mapping

---

## Media Gallery

![Top View](/fire-truck/top-view.png)
_Top-down view showing ArUco marker and electronics layout_

![Electronics Top View](/fire-truck/electronics-top-view.png)
_Electronics bay showing Arduino, motor driver, and wiring_

![Electrical Schematic Detail](/fire-truck/electrical-schematic2.png)
_Detailed schematic of power distribution and sensor connections_

📹 **[Full Demonstration Video](https://www.youtube.com/embed/XeLIS-Nzu3Y)**: Watch complete mission run

---

## Technologies Used

**Hardware**: Arduino Uno, HC-SR04 Ultrasonic Sensors, DHT20 Temperature Sensors, L298N Motor Driver, Standard Servo, 12V DC Motors  
**Software**: Arduino IDE, C++  
**Design**: SolidWorks 2021, Fusion 360  
**Manufacturing**: FDM 3D Printing (PETG), Laser Cutting (Plywood, Acrylic)  
**Testing**: Serial Monitor, Oscilloscope (motor signal debugging)

---

_This project represents my first complete robotics system—from concept to competition—teaching me the fundamentals of autonomous system design and integration._
