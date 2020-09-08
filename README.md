# Particle-Filter-SLAM
Project 2 ECE276A Project, UCSD, Winter-2020

## File Descriptions
SLAM.py: Performs the Particle filter SLAM algorithm to generate the MAP and the corresponding Robot trajectory

Transform.py: Library to evalute lidar to body and body to world homogenous tranformation matrix. This file is imported in the main SLAM.py file.

load_data.py: Python script to load the robot data i.e. Laser scans, joint angles, odometetry values etc.

Result: Folder containing the final Maps obtained.

Note: This code is a bit scrappy and very slow. It contains some useless lines of code (since I was in a hurry for submission). I will optimize the code very soon. Thanks for your patience :).
