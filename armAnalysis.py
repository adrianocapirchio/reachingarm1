# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:35:04 2018

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import utilities as utils
from arm import Arm





#width_table  = np.array([-0.4, 0.07]) 
#height_table = np.array([ 0.2, 0.6]) 


shoulderRange = np.deg2rad(np.array([-60.0, 150.0]))
elbowRange    = np.deg2rad(np.array([  0.0, 180.0])) 

arm = Arm(shoulderRange, elbowRange)



corner_point = 100


alpha_grid = utils.changeRange(np.linspace(0.0,1.0,corner_point), 0.0 , 1.0, shoulderRange[0], shoulderRange[1])
beta_grid = utils.changeRange(np.linspace(0.0,1.0,corner_point), 0.0 , 1.0, elbowRange[0], elbowRange[1])
rawGrid = np.meshgrid(alpha_grid,beta_grid)
grid = np.vstack(([rawGrid[0].T],[rawGrid[1].T])).T











goalRange = 0.03

goalList = [np.array([-0.07 -.05, 0.16 +0.04]),
            np.array([ 0.10 -.05, 0.16 +0.04]),
            np.array([-0.24 -.05, 0.16 +0.04]),
            np.array([-0.07 -.05, 0.42 +0.04]),        
            np.array([ 0.10 -.05, 0.42 +0.04]),
            np.array([-0.24 -.05, 0.42 +0.04])]






fig1   = plt.figure("arm range of motion", figsize=(8,8))
ax1    = fig1.add_subplot(111, aspect='equal')
ax1.set_xlim([-1.0,1.0])
ax1.set_ylim([-1.0,1.0])

arm1, = ax1.plot([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf], 'k-', color = 'pink', linewidth = 10)
hand1, = ax1.plot([arm.xEndEf], [arm.yEndEf], 'o', color = 'black' , markersize= 20)


for i in xrange(len(grid[0])):
    for j in xrange(len(grid[1])): 
        arm.theta1 = grid[i,j,:][0].copy()
        arm.theta2 = grid[i,j,:][1].copy()
        arm.getElbowPose()
        arm.getEndEffectorPose()
        ax1.scatter(arm.xEndEf,arm.yEndEf, c='dodgerblue', marker ='o', s = 10)


#table = patches.Rectangle(np.array([width_table[0], height_table[0]]), np.sqrt((width_table[0] - width_table[1])**2), np.sqrt((height_table[0] - height_table[1])**2) ,linewidth=1,edgecolor='black',facecolor='white')
#ax1.add_artist(table)

circle1 = plt.Circle((goalList[0]), goalRange, color = 'yellow') 
edgecircle1 = plt.Circle((goalList[0]), goalRange, color = 'black', fill = False) 
ax1.add_artist(circle1)
ax1.add_artist(edgecircle1)

circle2 = plt.Circle((goalList[1]), goalRange, color = 'yellow') 
edgecircle2 = plt.Circle((goalList[1]), goalRange, color = 'black', fill = False) 
ax1.add_artist(circle2)
ax1.add_artist(edgecircle2)

circle3 = plt.Circle((goalList[2]), goalRange, color = 'yellow') 
edgecircle3 = plt.Circle((goalList[2]), goalRange, color = 'black', fill = False) 
ax1.add_artist(circle3)
ax1.add_artist(edgecircle3)

circle4 = plt.Circle((goalList[3]), goalRange, color = 'yellow') 
edgecircle4 = plt.Circle((goalList[3]), goalRange, color = 'black', fill = False) 
ax1.add_artist(circle4)
ax1.add_artist(edgecircle4)

circle5 = plt.Circle((goalList[4]), goalRange, color = 'yellow') 
edgecircle5 = plt.Circle((goalList[4]), goalRange, color = 'black', fill = False) 
ax1.add_artist(circle5)
ax1.add_artist(edgecircle5)

circle6 = plt.Circle((goalList[5]), goalRange, color = 'yellow') 
edgecircle6 = plt.Circle((goalList[5]), goalRange, color = 'black', fill = False) 
ax1.add_artist(circle6)
ax1.add_artist(edgecircle6)

rewardCircle = plt.Circle((goalList[0]), goalRange, color = 'red')  
ax1.add_artist(rewardCircle)









