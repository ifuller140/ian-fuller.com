---
title: 'Aerial-Aquatic Manipulator'
description: 'Research into multi-domain drone with manipulation capabilities'
image: 'aam/aam.jpg'
preview: 'aam/preview.gif'
media: 'https://www.youtube.com/embed/ZA9tjAiyqa8?si=wBtlht-8Ws-gSZtj'
priority: 7
tags:
  - Research
  - NVIDIA Isaac Sim
  - 3D Printing
  - ROS
  - Mechanical Design
links:
  - text: Research Paper
    href: https://arxiv.org/pdf/2412.19744
  - text: GitHub
    href: https://github.com/konakarthik12/AAM-SEALS?tab=readme-ov-file
  - text: Project Website
    href: https://aam-seals.umd.edu/
  - text: OARL Lab Website
    href: https://yantianzha.github.io/oarl.github.io/
---

## Project Overview

Multi-domain robotics represents one of the most challenging frontiers in autonomous systems. The **Aerial-Aquatic Manipulator (AAM)** project, conducted under [Dr. Yantian Zha](https://yantianzha.github.io/) at the University of Maryland's **Omni-domain AI and Robotics Lab (OARL)**, aims to develop a drone capable of seamless air-to-water transitions while performing manipulation tasks in both environments.

I joined this research effort during the theoretical design phase and have since transitioned to **hardware development and manufacturing**, serving as the mechanical design lead responsible for translating simulation concepts into functional prototypes. This work has contributed to a published paper and an upcoming follow-up paper where I am **second author**.

![AAM Concept Render](/aam/concept-render.jpg)
_Initial concept design and fields of practical use cases_

---

## Research Motivation

Traditional drones excel in air or water, but rarely both. Applications like:

- **Marine biology research**: Tracking and sampling aquatic life
- **Search and rescue**: Operating across flood zones and open water
- **Infrastructure inspection**: Bridges, dams, and offshore structures
- **Environmental monitoring**: Coral reefs, water quality sampling

...all require platforms that can transition between domains without manual intervention. Our research addresses this gap with a system that can **fly, dive, swim, and manipulate** objects in both environments.

---

## My Role: From Paper to Prototype

### Phase 1: Conceptual Design & Initial Prototyping (2024)

I joined the team during the publication of our [initial research paper](https://arxiv.org/pdf/2412.19744), which outlined the theoretical framework for aerial-aquatic manipulation. My first task was to transform these concepts into manufacturable designs.

**Initial Contributions:**

- Designed and 3D printed the **first physical concept drone** used for paper demonstrations
- Developed waterproofing strategies for electronics enclosures
- Collaborated with PhD students to validate mechanical constraints in simulation

![First Prototype](/aam/first-prototype.jpg)
_Initial 3D printed concept drone used in the research paper_

**Key Design Constraint**: The manipulator needed to function effectively in both air (low resistance, lightweight) and water (high resistance, buoyancy compensation). This drove a **transformable gripper design** that changes configuration based on the medium.

### Phase 2: Advanced Development & Flight Testing (2024-Present)

With the theoretical foundation established, we moved to **functional hardware**. I now lead the mechanical design and manufacturing for both the drone platform and manipulation system.

**Current Responsibilities:**

- **Drone Airframe Design**: Designing the structural frame for a hybrid quadcopter/underwater vehicle
- **Manipulator Development**: Creating a 3-DOF robotic arm optimized for dual-domain operation
- **Manufacturing Oversight**: Managing 3D printing, machining, and assembly processes
- **Integration Testing**: Conducting flight tests and water trials to validate designs

![Current Drone Prototype](/aam/drone-flying.jpg)
_Latest prototype during flight testing—multiple successful flights achieved_

---

## Technical Deep Dive: Dual-Domain Challenges

### Challenge 1: Propulsion System Transition

**Air Mode Requirements:**

- High thrust-to-weight ratio (>2:1)
- Rapid response for stability
- Energy efficiency for extended flight time

**Water Mode Requirements:**

- Waterproofed motor enclosures
- Thrust vectoring for 6-DOF control
- Buoyancy compensation system

**Solution**: We use **brushless motors with conformal coating** and custom 3D-printed shrouds that serve dual purposes:

1. **In air**: Act as ducts to improve thrust efficiency
2. **In water**: Create sealed chambers preventing water ingress while allowing thrust transmission

![Propulsion System CAD](/aam/propulsion-cad.png)
_Dual-mode propulsion system with sealed motor housings_

### Challenge 2: Manipulator Design for Two Mediums

Designing a gripper that works in both air and water required balancing:

- **Weight** (must be light enough to not impair flight)
- **Strength** (must overcome water resistance during manipulation)
- **Dexterity** (needs sufficient DOF for object grasping)

**My Design Approach:**

I developed a **3-DOF compliant gripper** using:

- **3D printed TPU (flexible) fingers** that conform to object shapes
- **Servo-driven actuation** with waterproof housings
- **Force feedback** via current sensing on servos (detects grasp success)

The gripper uses different grasping strategies per domain:

- **Air**: Precision pinch grasp for small objects
- **Water**: Power grasp with increased closure force to overcome drag

![Manipulator CAD](/aam/manipulator-cad.png)
_3-DOF manipulator arm with compliant gripper design_

![Manipulator Testing](/aam/manipulator-test.gif)
_Gripper successfully grasping objects in water testing_

### Challenge 3: Transitioning Between Mediums

The air-to-water transition is the most critical phase. We needed to address:

1. **Impact Force**: Water entry creates significant shock loads
2. **Attitude Control**: Quadcopter control laws don't work underwater
3. **Buoyancy Management**: Drone must achieve neutral buoyancy upon submersion

**Solution Architecture**:

- **Foam-filled compartments** provide adjustable buoyancy
- **Control mode switching** detects water immersion via conductivity sensors
- **Dampened landing legs** absorb impact forces during water entry

The transition sequence:

1. Drone approaches water surface at controlled descent rate
2. Conductivity sensors detect immersion
3. Controller switches from quadcopter PID to underwater thruster control
4. Ballast system adjusts to achieve neutral buoyancy
5. Underwater navigation begins

<video
width="100%"
autoplay
loop
muted
className="rounded-lg shadow-lg"

>

  <source src="/aam/transition2.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>
_Simulation of realistic air-to-water transition_

---

## Simulation-First Development

Before any physical testing, we validate designs in **NVIDIA Isaac Sim**. This simulator provides:

- **Accurate physics modeling** for both air and water
- **Realistic sensor simulation** (cameras, IMUs, depth sensors)
- **Digital twin development** that matches hardware configurations exactly

I work closely with the simulation team to ensure mechanical designs are validated virtually before manufacturing, dramatically reducing prototype iteration cycles.

<video
width="100%"
autoplay
loop
muted
className="rounded-lg shadow-lg"

>

  <source src="/aam/crab-catch.mp4" type="video/mp4" />
  Your browser does not support the video tag.
</video>
_Isaac Sim water simulation showing manipulator interaction with marine environment_

**Workflow:**

1. I design mechanical components in **SolidWorks**
2. Export URDF models to Isaac Sim
3. Simulation team tests control algorithms
4. I iterate designs based on simulation results
5. Manufacture and validate in physical testing

This workflow has reduced our prototype iteration time from **weeks to days**.

---

## Manufacturing & Assembly

All custom components are manufactured in-house:

**3D Printing** (Primary Method):

- **Material**: PETG for water resistance and structural strength
- **Printer**: Prusa i3 MK3S+ with 0.2mm layer height
- **Post-processing**: Acetone vapor smoothing for waterproofing

**CNC Machining** (Critical Components):

- Aluminum mounting plates for motor assemblies
- Delrin bushings for manipulator joints (low friction, water-resistant)

**Electronics Integration**:

- Custom PCB for sensor integration (designed by electrical team)
- Waterproof connectors rated to IP68 (10m depth)
- Potted electronics in marine-grade epoxy

![Manufacturing Process](/aam/manufacturing-process.jpg)
_3D printing and assembly of drone components_

**Quality Control**: Every component undergoes:

- Dimensional verification with calipers
- Pressure testing in water tank (simulates 5m depth)
- Leak detection using vacuum chamber method

---

## Current Status & Results

### Achievements:

✅ **Multiple successful flight tests** with full payload  
✅ **Functional manipulator** tested in both air and water  
✅ **Published research paper** with 3D printed concept prototype  
✅ **Upcoming publication** as second author (in review)  
✅ **Stable air-to-water transitions** in controlled testing

### Performance Metrics:

- **Flight time**: 12 minutes (current battery configuration)
- **Underwater operation**: 8 minutes
- **Manipulator reach**: 45cm from drone center
- **Gripper force**: 5N in air, 3N in water (drag compensated)
- **Maximum depth**: 5 meters (current test limit)

![Flight Testing](/aam/flight-test.jpg)
_Field testing of the latest prototype_

### Team Collaboration

This is a **highly collaborative** research project. I work alongside:

**Principal Investigator**: Dr. Yantian Zha (Omni-domain AI and Robotics Lab)

**PhD Students**:

- Karthik Konakalla (Lead researcher, control systems)
- [Additional graduate students working on simulation and autonomy]

**My Focus**: Mechanical design, manufacturing, and hardware integration

**Their Focus**: Control algorithms, simulation, autonomy, and experimental validation

---

## Upcoming Work

As we prepare the next paper for submission, I'm focused on:

1. **Refined Manipulator Design**: Reducing weight by 30% while maintaining strength
2. **Sensor Integration**: Adding force-torque sensors to the manipulator for haptic feedback
3. **Autonomous Grasping**: Collaborating on computer vision pipeline for object detection underwater
4. **Extended Testing**: Open water trials in Chesapeake Bay (planned for Spring 2025)

---

## Technical Challenges & Lessons Learned

**Biggest Challenge**: Waterproofing while maintaining functionality.  
**Lesson Learned**: Over-engineering seals is essential—every test reveals new leak paths. We now test all components independently before system integration.

**Most Surprising Issue**: Buoyancy is harder to control than expected. Small changes in component density compound quickly. We now maintain a detailed mass budget for every design iteration.

**Key Insight**: Simulation is invaluable, but real-world testing always reveals unexpected behaviors. The transition between simulation and hardware requires disciplined iteration and documentation.

---

## Research Impact

This work contributes to:

- **Expanding the operational envelope** of autonomous systems
- **Enabling new environmental monitoring** capabilities
- **Advancing multi-domain robotics** research
- **Bridging the sim-to-real gap** through rigorous validation

Our upcoming paper will be the first to demonstrate a **fully functional aerial-aquatic manipulator** with validated performance in both domains.

---

## Technologies & Tools

**Mechanical Design**: SolidWorks, Fusion 360  
**Simulation**: NVIDIA Isaac Sim, Gazebo, ROS  
**Manufacturing**: FDM 3D Printing (Prusa i3), CNC Mill  
**Materials**: PETG, TPU, Aluminum 6061, Delrin  
**Electronics**: Custom PCBs, Pixhawk flight controller, Raspberry Pi  
**Testing**: Pressure chamber, water tank, motion capture system

---

_This research is ongoing. For the latest updates, visit the [lab website](https://aam-seals.umd.edu/) or follow our [GitHub repository](https://github.com/konakarthik12/AAM-SEALS)._
