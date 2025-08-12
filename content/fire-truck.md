---
title: 'Autonomous Fire Truck'
description: 'A fully autonomous robot to extinguish candles while traversing difficult terrain'
image: 'fire-truck/fire-truck.jpg'
preview: 'fire-truck/preview.gif'
priority: 3
tags:
  - Arduino
  - SolidWorks
  - 3D Printing
---

## Overview

I was tasked with the mission of designing, building and coding an autonomous robot to complete 4 core objectives:

- **Topography Sensing:** Determine and transmit the orientation of a block containing candles at varying heights
- **Fire Sensing:** Determine and transmit the number of lit candles on the block
- **Fire Suppresion:** Extinguish all candles except for the center candle
- **Localization and Navigation:** Use a camera system to determine position and path plan to the desired target location

This here is my final design of the autonomous fire extinguishing robot
![Fire Truck](/fire-truck/fire-truck.png)

## Topography Sensing

![Topography Sensing](/fire-truck/topography.png)

Two HC-SR04 ultrasonic distance sensors are placed at the front of the robot. A custom, 3D printed mount was made to place the distance sensors at the correct heights. The strategy used for this mission objective was to take advantage of the differing heights for the center and right portions of each side of the candle block. The distance sensors are used to determine the distance between the robot and the center and right portions of the block. Conditionals are used as follows: if the center portion is closer than the right portion, the side is A. If the center portion is further than the right portion, the side is B. If the center portion is the same distance as the right portion, the side is C. An image of the each block side is shown below, as well as the distance sensors.
![Ultrasonic Sensors](/fire-truck/ultrasonic-sensors.png)

## Fire Sensing

To be able to detect if a flame is present I added a DHT20 temperature and humidity sensor to each suppression node. The temperature sensing was utilized for this mission objective. If the temperature was greater than a certain threshold above room tempurature, it was assumed there was a candle under the node supplying the heat, therefore the number of lit candles can be increased by one. The sensor in the node is shown below.
![Temp Sensor](/fire-truck/temp-sensor.png)

## Fire Suppresion

This mission objective was the most challenging of all three. The approach used was a plywood platform that was able to move in a folding motion, retracted over the chassis and extended over the candle block. On the platform, there are four 3D printed suppression nodes that are wrapped in foil and hung from the platform by 3D printed rods. With larger holes, these rods allow the nodes to conform to the varying heights of the candles. In the center of the platform, there is a hole to ensure that the center candle is not put out. This platform was moved by a servo and a custom servo arm. The servo arm has holes that zip-ties can go through, and those zip-ties are attached to two of four acrylic beams that support the upper platform. Eight ball bearings in total allow for the acrylic beams to rotate smoothy. Below are images of the platform, the nodes, and the retracted and extended positions.
![System Pipeline](/fire-truck/suppression-nodes.png)
![System Pipeline](/fire-truck/suppression-nodes-extended.png)
![System Pipeline](/fire-truck/suppression-nodes-retracted.png)

## Localization and Navigation

The robot uses an Aruco marker position strategically on the top of the robot to be able to use the external overhead camera system above the arena to get an accurate position of my robot at all times. With this position and orientation data I set up a control system to guide the robot through its fire extinguishing objective. Here is how I implemented the control algorithm.
![System Pipeline](/fire-truck/top-view.png)
![System Pipeline](/fire-truck/pipeline.png)

## Propulsion

![Full View](/fire-truck/full-body-view-extended-cad.png)

This robot is two wheel drive. Its two back wheels are motorized, while the front two wheels are caster wheels. The back wheels are 3.15" in diameter, which is large enough to traverse the textured flooring of the arena, but not large enough to make it over the hump or the rumble pad obstacle. The front caster wheels are on steppers to match the height of the back wheels. The motors are powered by a 12V battery, and they are controlled by an L298N motor driver, which is connected to the Arduino Uno. Images of the back wheels, front wheels, and motor driver are shown below.

![Full View](/fire-truck/full-body-view-extended-cad.png)

## Electronics

![Electronics Top View](/fire-truck/electronics-top-view.png)

The robot uses an Arduino Uno as its microcontroller. It is powered by a 12V battery, shared with the two drive motors. Connected to the Arduino is two ultrasonic distance sensors, four temperature sensors, a WiFi module, the motor driver, and the servo. Notable features about the wiring and electronics include the split power source on the 12V battery, for the motors and Arduino, each having their own kill switch. Also, all of the analog pins on the Arduino are occupied, and there is one open PWM pin. Here you can see images of the complete Arduino wiring including the electrical schematics.

![Electrical Schematic](/fire-truck/electrical-schematic.png)
![Electrical Schematic](/fire-truck/electrical-schematic2.png)

## Final Run
