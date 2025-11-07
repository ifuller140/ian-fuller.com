---
title: 'Terraformers - Mars Rover'
description: 'Competition rover for University Rover Challenge'
image: 'terraformers/terraformers.jpg'
preview: 'terraformers/preview.mp4'
media: 'https://www.youtube.com/embed/w50tzLyhFUw?si=VD4bZxoAxfcRkvur&amp;controls=0'
priority: 8
tags:
  - Mechanical Design
  - 3D Printing
  - SolidWorks
  - Manufacturing
  - Team Project
---

## Competition Overview

The **[University Rover Challenge (URC)](http://urc.marssociety.org/)** is the world's premier robotics competition for college students, challenging teams to design and build the next generation of Mars rovers. Hosted annually in the Utah desert, URC simulates the harsh Martian environment where rovers must perform:

- **Autonomous navigation** across rocky terrain
- **Equipment servicing** (opening hatches, flipping switches, turning valves)
- **Science missions** (soil collection, life detection experiments)
- **Extreme retrieval & delivery** (transporting objects across obstacles)

As a member of **Robotics @ Maryland's Terraformers team**, I served as **Manufacturing Lead**, responsible for coordinating the design-for-manufacturing process and overseeing production of all 3D printed and machined components.

![Terraformers Rover](/terraformers/rover-full-view.jpg)
_Terraformers rover during field testing in Maryland_

---

## My Role: Design for Manufacturing (DFM)

Building a competition-grade rover isn't just about clever designs—it's about ensuring every component can be **reliably manufactured** with available resources. As Manufacturing Lead, I bridged the gap between CAD models and physical parts.

### Responsibilities

**1. Design Review & Optimization**

- Reviewed all SolidWorks assemblies before production approval
- Identified manufacturability issues (unsupported overhangs, excessive support material, wall thickness violations)
- Proposed design modifications to improve printability and strength

**2. Manufacturing Coordination**

- Managed production schedule across 10+ 3D printers
- Prioritized critical components (drivetrain > arm > sensors)
- Tracked completion status and quality issues

**3. Team Training**

- Instructed designers on DFM principles (print orientation, support strategies, tolerance stacking)
- Created documentation on best practices for FDM printing
- Taught post-processing techniques (heat-set inserts, acetone smoothing)

**4. Quality Assurance**

- Inspected finished parts for dimensional accuracy
- Coordinated rework when parts failed fit checks
- Maintained inventory of spare components

![Manufacturing Workflow](/terraformers/manufacturing-workflow.png)
_Design-to-manufacturing pipeline I established for the team_

---

## Design for Additive Manufacturing (DFM Principles)

### Challenge: 3D Printing at Scale

Our rover required **200+ custom 3D printed parts**. At 4-10 hours per part, inefficient designs would blow our 3-month build schedule. I implemented systematic DFM reviews to optimize every component.

### Key Optimization Strategies

**1. Print Orientation Optimization**

Example: Robotic arm joint housing

- **Original design**: Required extensive support material (18hr print, 60% material waste)
- **My optimization**: Reoriented part to minimize overhangs (12hr print, 15% waste)
- **Result**: 33% time savings, $40 saved per part × 8 joints = $320 total

![Print Orientation Example](/terraformers/print-orientation.png)
_Before and after optimization—support material highlighted in red_

**2. Stress-Based Part Orientation**

3D printed parts are **anisotropic**—stronger in XY plane than Z direction (layer adhesion is weaker).

Example: Wheel motor mount

- **Critical load**: Torsional force from motor
- **My decision**: Orient part so torsion stresses stay in XY plane
- **Result**: Mount survived 50+ hours of testing without failure

```
Load Direction Analysis:
┌─────────────────────────────────┐
│  Force  │  Orientation Strategy │
├─────────────────────────────────┤
│ Tension │ Load along XY layers  │
│ Bending │ Neutral axis in Z     │
│ Torsion │ Shear in XY plane     │
└─────────────────────────────────┘
```

**3. Tolerance Accommodation**

FDM printers have ±0.2mm dimensional accuracy. I established design rules:

- **Press-fit holes**: Design 0.3mm undersized (sanding to final fit)
- **Clearance fits**: Design 0.4mm oversized (prevent binding)
- **Threaded inserts**: Use heat-set brass inserts (M3, M4, M5) instead of tapping plastic

![Tolerance Guidelines](/terraformers/tolerance-guide.png)
_Reference chart provided to design team_

**4. Support-Free Overhangs**

Unsupported overhangs >45° fail or require support removal.

Design modifications:

- Added fillets to transition from vertical to horizontal surfaces
- Redesigned mounting brackets with self-supporting geometry
- Split large parts into multiple pieces (print separately, assemble with fasteners)

---

## Manufacturing Process Management

### Production Pipeline

```
Design Freeze → DFM Review → Slice & Print → Post-Process → QA Check → Assembly
     ↓              ↓            ↓              ↓            ↓          ↓
  SolidWorks    My Review    PrusaSlicer    Me/Team      Calipers   Integration
```

### 3D Printing Infrastructure

**Equipment**:

- 8× Prusa i3 MK3S+ (primary workhorses)
- 2× Creality CR-10 (large format parts)
- 1× Ultimaker S5 (high-detail components)

**Materials**:

- **PETG**: Primary structural material (outdoor UV resistance, good strength)
- **PLA**: Prototyping only (not used on final rover)
- **TPU**: Flexible components (wheel treads, cable management)
- **ASA**: UV-critical parts (sensor housings exposed to sunlight)

![Print Farm](/terraformers/print-farm.jpg)
_Our 10-printer manufacturing setup in the lab_

### Post-Processing Techniques

Every part underwent finishing:

**1. Support Removal**: Needle-nose pliers + flush cutters
**2. Sanding**: 120→220→400 grit progression for smooth surfaces
**3. Heat-Set Inserts**: Soldering iron at 220°C to embed brass threaded inserts
**4. Acetone Vapor Smoothing** (for ASA parts): Improves surface finish and layer bonding

---

## Key Components I Manufactured

### 1. Drivetrain Components

The rover's drivetrain required the highest precision—any wobble compounds across 6 wheels.

**Parts produced**:

- Wheel hubs (6×, PETG)
- Motor mounts (6×, PETG)
- Suspension arms (12×, PETG + carbon fiber rods)

**Challenge**: Ensuring concentric bearing seats  
**Solution**: Printed with 0.6mm nozzle for dimensional stability, reamed bearing bores to exact tolerance with hand reamer

![Drivetrain Assembly](/terraformers/drivetrain-assembly.jpg)
_3D printed wheel assembly with bearing integration_

### 2. Robotic Arm Linkages

5-DOF arm required lightweight yet strong linkages.

**Material choice**: PETG with 40% infill (strength-to-weight optimization)

**Design collaboration**: Worked with arm subteam to reduce mass by 30% through topology optimization

- Removed non-load-bearing material
- Added internal ribbing for torsional rigidity
- Validated in SolidWorks FEA before printing

![Arm Linkage](/terraformers/arm-linkage.jpg)
_Optimized arm linkage with internal ribbing structure_

### 3. Sensor Housings

All electronics needed protection from dust and impact.

**Requirements**:

- Dust ingress protection (IP54 rating)
- Ventilation for heat dissipation
- Easy access for repairs

**My design improvements**:

- Added snap-fit covers (faster field repairs than screws)
- Incorporated gasket channels (sealed with foam tape)
- Used ASA material (UV stable for outdoor testing)

---

## Non-Metal Machined Parts

In addition to 3D printing, I coordinated production of machined plastic components:

**Delrin (Acetal) Parts**:

- Gearbox housings (low friction, wear resistant)
- Pulley bushings (smooth rotation, dimensional stability)

**Acrylic Parts**:

- Electronic enclosure panels (laser cut)
- Sensor windows (transparency needed)

**Manufacturing Process**:

- Designed parts in SolidWorks
- Generated 2D drawings with tolerances
- Coordinated with university machine shop for CNC milling
- Inspected finished parts with calipers and micrometers

![Machined Parts](/terraformers/machined-parts.jpg)
_Delrin gearbox components_

---

## Challenges & Problem-Solving

### Challenge 1: Part Warping

**Problem**: Large flat parts (chassis panels) warped during printing, causing assembly issues.

**Root cause**: Uneven cooling creates internal stresses

**My solution**:

- Increased bed temperature (85°C for PETG)
- Added brim (wider base adhesion)
- Enclosed printer to reduce temperature gradients
- Redesigned parts with ribbing to resist warping

**Result**: Warp reduced from 2mm to <0.3mm

### Challenge 2: Print Failures Mid-Job

**Problem**: 10-hour prints failing at hour 8 due to filament tangles or power outages.

**Solutions implemented**:

- Installed filament runout sensors on all printers
- Added UPS (uninterruptible power supply) to critical printers
- Implemented resume-from-failure in firmware
- Required team members to check on prints every 2 hours

**Result**: Failure rate dropped from 15% to 3%

### Challenge 3: Tight Competition Deadline

**Problem**: Major design change 4 weeks before competition (arm redesign)

**My response**:

- Prioritized critical path components (arm linkages before cosmetic covers)
- Ran printers 24/7 with staggered scheduling
- Recruited additional team members to monitor prints overnight
- Expedited QA checks (accepted minor cosmetic flaws if functionally sound)

**Outcome**: Delivered all parts on time, rover assembly completed with 3 days to spare

---

## Competition Performance

While we **did not advance past the qualifying round**, the manufacturing process was a success:

✅ **Zero mechanical failures** during qualification videos  
✅ **All 200+ printed parts** delivered on schedule  
✅ **Assembly completed** without major fit issues  
✅ **Valuable lessons learned** for future team iterations

**What We Learned**:

- Our **software integration** needed more testing time (we focused too much on hardware)
- **Field testing earlier** would have revealed control system issues
- **Manufacturing pipeline worked well**—next year's team adopted my DFM process

![Team at Competition](/terraformers/team-photo.jpg)
_Terraformers team at University Rover Challenge qualifying round_

---

## Impact on Team

My manufacturing leadership established systems still used by the team:

**Documentation Created**:

- DFM design guidelines (20-page handbook)
- Print orientation decision tree
- Material selection flowchart
- Post-processing SOPs (standard operating procedures)

**Training Delivered**:

- Trained 15+ team members on 3D printing
- Taught SolidWorks best practices for printability
- Mentored new manufacturing lead for next year's team

**Process Improvements**:

- Reduced average part iteration cycles from 3 to 1.5
- Cut print failure rate by 80%
- Decreased material waste by 40%

---

## Technical Skills Developed

**Design Tools**:

- SolidWorks (assemblies, drawings, FEA)
- PrusaSlicer (slicing optimization)
- Meshmixer (support structure editing)

**Manufacturing**:

- FDM 3D printing (multi-material, multi-printer)
- Post-processing techniques
- Quality inspection (calipers, micrometers)

**Project Management**:

- Production scheduling
- Resource allocation (printer time, material budget)
- Risk mitigation (backup parts, spare capacity)

**Communication**:

- Cross-functional collaboration (mechanical, electrical, software teams)
- Technical documentation
- Training and mentorship

---

## Lessons Learned

**Manufacturing isn't just making parts**—it's about:

- Understanding constraints (time, budget, equipment capabilities)
- Communicating with designers (iterative feedback loop)
- Managing risk (spare parts, backup plans)
- Balancing perfectionism with pragmatism (good enough to compete beats perfect but late)

**Key Insight**: The best designs are useless if you can't manufacture them reliably and on schedule. DFM must be integrated into the design process from day one, not bolted on at the end.

---

## Future Improvements

If I were to lead manufacturing again:

1. **Start DFM reviews earlier** (during initial sketches, not after full CAD)
2. **Implement version control** for physical parts (track which revision is on the rover)
3. **Build more spares** of high-failure-risk components
4. **Invest in better printer enclosures** (temperature control reduces failures)

---

## Technologies & Tools

**CAD**: SolidWorks 2022, Fusion 360  
**Slicing**: PrusaSlicer, Cura  
**3D Printers**: Prusa i3 MK3S+, Creality CR-10, Ultimaker S5  
**Materials**: PETG, ASA, TPU, PLA  
**Machining**: CNC Mill (coordinated with machine shop)  
**Inspection**: Digital calipers, micrometers, gauge blocks

---

_Manufacturing Lead, R @ M Terraformers Team, URC 2023_
