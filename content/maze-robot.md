---
title: 'LiDAR Maze Robot'
description: 'Autonomous maze solving robot utilizing LiDAR and A* algorithm'
image: 'maze-robot/maze-robot.jpg'
preview: 'maze-robot/preview.gif'
media: 'https://www.youtube.com/embed/srvkYn-s5TI?si=PAD8qaLIVJfSjBxT&amp;start=4'
priority: 1
tags:
  - Python
  - ROS
  - TurtleBot
  - A* Algorithm
  - LiDAR
links:
  - text: View on GitHub
    href: https://github.com/ifuller140/maze-robot
---

## Project Overview

Autonomous navigation through unknown environments is a fundamental challenge in robotics. For this project, I developed a complete navigation stack for a **TurtleBot3 Waffle** that explores and solves mazes using **LiDAR sensing** and **A\* pathfinding**. The robot builds a map in real-time, plans optimal paths, and executes movements—all without any prior knowledge of the maze layout.

This project demonstrates the integration of **perception** (LiDAR processing), **planning** (A\* search), and **control** (trajectory execution) within the **ROS** framework—core competencies for any autonomous robotics system.

![Maze Solving in Action](/maze-robot/maze-solving.gif)
_TurtleBot autonomously navigating through maze_

---

## Technical Challenge

The robot must:

1. **Map the environment** using only 360° LiDAR scans
2. **Localize itself** within the partially-built map
3. **Plan a path** to unexplored areas or the goal
4. **Execute motion** while avoiding obstacles
5. **Handle dynamic updates** as new walls are discovered

All of this happens **autonomously** in real-time with no human intervention.

---

## System Architecture

### Hardware Platform: TurtleBot3 Waffle

The TurtleBot3 is a differential drive robot equipped with:

- **LDS-01 LiDAR**: 360° laser scanner (1,800 samples/rotation, 12m range)
- **Raspberry Pi 4**: Main computer running ROS
- **OpenCR Board**: Motor control and IMU
- **Differential Drive**: Two independently controlled wheels

![TurtleBot3 Hardware](/maze-robot/turtlebot-hardware.jpg)
_TurtleBot3 Waffle with LiDAR sensor_

### Software Stack

```
┌─────────────────────────────────────┐
│     A* Path Planning Node           │
│  (Computes optimal path to goal)    │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     Navigation Controller            │
│  (Executes path with obstacle avoid)│
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     SLAM Node (GMapping)             │
│  (Builds map from LiDAR scans)       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│     LiDAR Driver (lds_driver)        │
│  (Publishes laser scan data)         │
└──────────────────────────────────────┘
```

**ROS Topics & Data Flow**:

- `/scan`: LiDAR publishes laser range data
- `/map`: GMapping publishes occupancy grid
- `/cmd_vel`: My controller publishes velocity commands
- `/odom`: TurtleBot publishes odometry (position estimate)

---

## Mapping & Localization: SLAM

The robot uses **GMapping** (Grid-based FastSLAM) for simultaneous localization and mapping. This algorithm:

1. **Takes LiDAR scans** as input
2. **Estimates robot pose** using particle filter
3. **Updates occupancy grid** (which cells are walls, free space, or unknown)

**Why GMapping?** It's computationally efficient enough to run in real-time on a Raspberry Pi while producing accurate maps.

**Occupancy Grid Representation**:

- **0**: Definitely free space (robot can drive here)
- **100**: Definitely occupied (wall detected)
- **-1**: Unknown (not yet explored)

![Occupancy Grid Map](/maze-robot/occupancy-grid.png)
_Real-time occupancy grid map being built as robot explores_

---

## Path Planning: A\* Algorithm Implementation

Once the map exists, the robot needs to find the optimal path from its current position to a goal. I implemented **A\* (A-star)** search, which guarantees finding the shortest path if one exists.

### Algorithm Overview

A\* uses a heuristic function to guide search toward the goal:

```
f(n) = g(n) + h(n)
```

Where:

- `g(n)` = cost from start to node n (actual distance traveled)
- `h(n)` = heuristic estimate from n to goal (Euclidean distance)
- `f(n)` = total estimated cost of path through n

### My Implementation

```python
def a_star_search(start, goal, occupancy_grid):
    """
    A* pathfinding on occupancy grid
    Returns: List of waypoints from start to goal
    """
    open_set = PriorityQueue()
    open_set.put((0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, occupancy_grid):
            # Skip if neighbor is occupied
            if occupancy_grid[neighbor] > 50:  # 50% threshold
                continue

            tentative_g = g_score[current] + distance(current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                open_set.put((f_score[neighbor], neighbor))

    return None  # No path found

def heuristic(pos1, pos2):
    """Euclidean distance heuristic"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
```

**Key Optimization**: I use a **priority queue** (min-heap) to efficiently retrieve the node with lowest f-score, ensuring O(log n) operations instead of O(n) for a naive list search.

### Path Smoothing

Raw A\* paths follow grid cell centers, creating jagged, inefficient trajectories. I implemented **path smoothing**:

1. **Shortcutting**: Check if intermediate waypoints can be skipped (straight-line path is collision-free)
2. **Spline interpolation**: Fit smooth curves through remaining waypoints

```python
def smooth_path(path, occupancy_grid):
    """
    Remove unnecessary waypoints using line-of-sight checks
    """
    if len(path) < 3:
        return path

    smoothed = [path[0]]
    current_idx = 0

    while current_idx < len(path) - 1:
        # Try to skip ahead as far as possible
        for skip_idx in range(len(path) - 1, current_idx, -1):
            if is_line_clear(path[current_idx], path[skip_idx], occupancy_grid):
                smoothed.append(path[skip_idx])
                current_idx = skip_idx
                break

    return smoothed
```

![Path Smoothing](/maze-robot/path-smoothing.png)
_Before and after path smoothing (red = raw A_, blue = smoothed)\*

---

## Motion Control

Having a path is one thing—executing it smoothly is another. I implemented a **pure pursuit controller** that:

1. **Looks ahead** along the path by a fixed distance
2. **Computes steering** to aim toward that lookahead point
3. **Adjusts speed** based on path curvature (slow down for turns)

```python
def pure_pursuit_control(robot_pose, path, lookahead_distance):
    """
    Pure pursuit algorithm for path following
    """
    # Find lookahead point on path
    lookahead_point = find_lookahead_point(robot_pose, path, lookahead_distance)

    # Compute desired heading
    dx = lookahead_point.x - robot_pose.x
    dy = lookahead_point.y - robot_pose.y
    desired_heading = math.atan2(dy, dx)

    # Compute heading error
    heading_error = normalize_angle(desired_heading - robot_pose.theta)

    # Control law
    linear_velocity = max_speed * (1.0 - abs(heading_error) / math.pi)
    angular_velocity = k_angular * heading_error

    return linear_velocity, angular_velocity
```

**Why Pure Pursuit?** It's simple, robust, and handles curved paths well. More sophisticated controllers (MPC, DWA) exist but add complexity without significant benefit for this application.

---

## Exploration Strategy

In unknown mazes, the robot must **explore intelligently** to find the goal. I implemented **frontier-based exploration**:

1. **Identify frontiers**: Boundaries between known free space and unknown areas
2. **Rank frontiers**: Prioritize by distance from robot (closer = explore first)
3. **Navigate to frontier**: Use A\* to plan path
4. **Repeat**: Once frontier reached, identify new frontiers and continue

```python
def find_frontiers(occupancy_grid):
    """
    Find frontier cells (free cells adjacent to unknown cells)
    """
    frontiers = []

    for x in range(grid_width):
        for y in range(grid_height):
            if occupancy_grid[x, y] == 0:  # Free cell
                # Check if any neighbor is unknown
                for neighbor in get_neighbors((x, y)):
                    if occupancy_grid[neighbor] == -1:  # Unknown
                        frontiers.append((x, y))
                        break

    # Cluster nearby frontier cells
    frontier_clusters = cluster_frontiers(frontiers, cluster_radius=0.5)

    return frontier_clusters
```

![Frontier Exploration](/maze-robot/frontier-exploration.png)
_Green = current frontiers, red = exploration path_

---

## Obstacle Avoidance

Even with a good plan, unexpected obstacles (or map errors) can appear. I added a **reactive obstacle avoidance layer**:

- Monitor LiDAR for obstacles within safety radius (0.3m)
- If obstacle detected, temporarily override path following
- Execute evasive maneuver (rotate away from obstacle)
- Resume path following once clear

This ensures the robot never collides, even if the map is slightly inaccurate.

---

## Performance & Results

### Maze Solving Success Rate

Tested on 5 different maze configurations:

- **5/5 mazes solved** successfully
- **Average solve time**: 4 minutes 23 seconds
- **Map accuracy**: 96% (compared to ground truth)

### Path Efficiency

Compared to wall-following (baseline algorithm):

- **32% shorter paths** on average
- **45% faster completion** (A\* finds direct routes, wall-following explores redundantly)

![Performance Comparison](/maze-robot/performance-graph.png)
_A_ vs. wall-following: path length and time comparison\*

### Real-World Testing

![Maze Test Arena](/maze-robot/test-arena.jpg)
_Physical maze setup used for testing_

**Challenges Overcome**:

- **LiDAR noise**: Added median filtering to smooth scan data
- **Wheel slippage**: Incorporated IMU data for better odometry
- **Narrow passages**: Tuned safety margins to allow passage while avoiding collisions

---

## Technical Challenges & Solutions

### Challenge 1: Grid Resolution Trade-off

**Problem**: Fine grids (1cm cells) are accurate but computationally expensive for A\*. Coarse grids (10cm cells) are fast but miss narrow passages.

**Solution**: Multi-resolution mapping:

- Use 2cm grid for mapping (capture detail)
- Downsample to 5cm grid for pathfinding (balance speed and accuracy)

### Challenge 2: Dynamic Map Updates

**Problem**: As the robot explores, the map changes. Previously planned paths may become invalid (newly discovered walls).

**Solution**: **Replanning trigger**:

- Monitor map for significant changes near planned path
- If change detected, abort current path and replan
- Implements safety check: only replan if not in narrow corridor (prevents oscillation)

### Challenge 3: Goal Reached Detection

**Problem**: Odometry drift means robot might never think it reached goal (position estimate is inaccurate).

**Solution**: **Fuzzy goal threshold**:

- Consider goal reached if within 0.2m radius
- Also check LiDAR to confirm goal area matches expected layout

---

## Code Structure

```
maze_robot/
├── launch/
│   └── maze_solver.launch       # ROS launch file
├── src/
│   ├── mapping_node.py          # GMapping wrapper
│   ├── planner_node.py          # A* implementation
│   ├── controller_node.py       # Pure pursuit control
│   └── explorer_node.py         # Frontier detection
├── config/
│   ├── gmapping_params.yaml     # SLAM parameters
│   └── controller_params.yaml   # Control gains
└── README.md
```

**ROS Integration**:

- Each component is a separate node for modularity
- Nodes communicate via topics (loose coupling)
- Parameters configurable via YAML (no recompilation needed)

---

## What I Learned

**Technical Skills**:

- Implementing classic search algorithms (A\*) for real-time robotics
- Working with occupancy grids and spatial representations
- Tuning control systems for smooth motion
- Debugging ROS multi-node systems

**Robotics Concepts**:

- SLAM is harder than it looks (odometry drift, loop closure)
- Planning is fast, but replanning intelligently is the real challenge
- Sensor noise is inevitable—design for robustness, not perfection

**Engineering Practices**:

- Modular design pays off (swapped A\* for Dijkstra in 10 minutes for comparison)
- Testing in simulation first (Gazebo) saved hours of hardware debugging
- Logging and visualization (RViz) are essential for understanding robot behavior

---

## Future Improvements

If I were to extend this project:

1. **Multi-robot coordination**: Multiple TurtleBots exploring collaboratively
2. **Dynamic obstacles**: Handle moving obstacles (other robots, people)
3. **3D mapping**: Use depth camera for multi-level mazes
4. **Machine learning**: Replace handcrafted control with learned policies (DRL)

---

## Media Gallery

![Maze Solution Visualization](/maze-robot/solution-visualization.png)
_Final solved maze with path overlay_

![RViz Visualization](/maze-robot/rviz-screenshot.png)
_Live RViz view during exploration_

📹 **[Demo Video](https://www.youtube.com/embed/srvkYn-s5TI)**: Watch the full maze solving run

---

## Technologies Used

**Robotics Framework**: ROS (Noetic)  
**SLAM**: GMapping  
**Pathfinding**: A\* algorithm (custom implementation)  
**Control**: Pure Pursuit controller  
**Programming**: Python 3.8  
**Simulation**: Gazebo  
**Visualization**: RViz  
**Hardware**: TurtleBot3 Waffle, LDS-01 LiDAR

---

_This project demonstrates fundamental autonomous navigation capabilities applicable to warehouse robots, delivery robots, and exploration rovers._
