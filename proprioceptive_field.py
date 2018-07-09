# -*- coding: utf-8 -*-
"""
Created on Mon Jul 02 15:12:25 2018

@author: Alex
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D

import utilities as utils
from arm import Arm




shoulderRange = np.deg2rad(np.array([-60.0, 150.0]))
elbowRange    = np.deg2rad(np.array([  0.0, 180.0])) 

arm = Arm(shoulderRange, elbowRange)


x_N = 41
y_N = 41


sigma = 1.0 / ( 2 * (x_N -1))


angles_state = np.zeros(x_N*y_N)

alpha_grid = utils.changeRange(np.linspace(0.0,1.0,x_N), 0.0 , 1.0, shoulderRange[0], shoulderRange[1])
beta_grid = utils.changeRange(np.linspace(0.0,1.0,y_N), 0.0 , 1.0, elbowRange[0], elbowRange[1])
rawGrid = np.meshgrid(alpha_grid,beta_grid)
grid = np.vstack(([rawGrid[0].T],[rawGrid[1].T])).T



x_grid = np.linspace(0.0,1.0,x_N)
y_grid = np.linspace(0.0,1.0,y_N)
rawXYGrid = np.meshgrid(x_grid,y_grid)
XYgrid = np.vstack(([rawGrid[0].T],[rawGrid[1].T])).T

proprioceptive_grid = utils.buildGrid(0.0,1.0,x_N,0.0,1.0,y_N).reshape(2,x_N*y_N)





goalRange = 0.03

goalList = [np.array([-0.07 -.05, 0.16 +0.04]),
            np.array([ 0.10 -.05, 0.16 +0.04]),
            np.array([-0.24 -.05, 0.16 +0.04]),
            np.array([-0.07 -.05, 0.42 +0.04]),        
            np.array([ 0.10 -.05, 0.42 +0.04]),
            np.array([-0.24 -.05, 0.42 +0.04])]






fig1   = plt.figure("proprioceptive_activation", figsize=(8,8))

ax1    = fig1.add_subplot(121, aspect='equal')
ax1.set_xlim([-1.0,1.0])
ax1.set_ylim([-1.0,1.0])

arm1, = ax1.plot([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf], 'k-', color = 'pink', linewidth = 10)
hand1, = ax1.plot([arm.xEndEf], [arm.yEndEf], 'o', color = 'black' , markersize= 20)


ax2    = fig1.add_subplot(122, projection ='3d', aspect='equal')
ax2.set_xlim([ 0.0, 1.0])
ax2.set_ylim([ 0.0, 1.0])
ax2.set_zlim([ 0.0, 1.0])
ax2.set_xlabel(r'$\alpha$')
ax2.set_ylabel(r'$\beta$')
ax2.set_zlabel(r'$\chi$')

norm_Arm_angles =np.array([utils.changeRange(arm.theta1,shoulderRange[0],shoulderRange[1],0.0,1.0), utils.changeRange(arm.theta2,elbowRange[0],elbowRange[1],0.0,1.0)]) 


angles_state = utils.rbf2D(norm_Arm_angles, angles_state,proprioceptive_grid,sigma)
angles_activation = np.dstack([proprioceptive_grid[1], proprioceptive_grid[0], angles_state])
angles_activation = angles_activation[0]

activation = ax2.plot_trisurf(angles_activation[:,1], angles_activation[:,0], angles_activation[:,2], cmap='copper',linewidth=0)
fig1.colorbar(activation)

theta1 =utils.changeRange(arm.xEndEf, shoulderRange[0], shoulderRange[1], 0.0, 1.0)
theta2 =utils.changeRange(arm.yEndEf, elbowRange[0], elbowRange[1], 0.0, 1.0)
arm_angles = np.array([theta1,theta2])

for i in xrange(len(grid[0])):
    for j in xrange(len(grid[1])): 
        arm.theta1 = grid[i,j,:][0].copy()
        arm.theta2 = grid[i,j,:][1].copy()
        arm.getElbowPose()
        arm.getEndEffectorPose()
        
        goal_3sigma = patches.Wedge(np.array([arm.xEndEf,arm.yEndEf]), 3 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.3, lw = 1)
        goal_2sigma = patches.Wedge(np.array([arm.xEndEf,arm.yEndEf]), 2 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.6, lw = 1)        
        goal_1sigma = patches.Wedge(np.array([arm.xEndEf,arm.yEndEf]), 1 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 1.0, lw = 1)
    
        ax1.add_artist(goal_3sigma)
        ax1.add_artist(goal_2sigma)
        ax1.add_artist(goal_1sigma)
        

       # ax1.scatter(arm.xEndEf,arm.yEndEf, c='dodgerblue', marker ='o', s = 10)
        
for i in xrange(len(grid[0])):
    for j in xrange(len(grid[1])):
        
        arm.theta1 = grid[i,j,:][0].copy()
        arm.theta2 = grid[i,j,:][1].copy()
        arm.getElbowPose()
        arm.getEndEffectorPose()
        
        theta3 =utils.changeRange(arm.xEndEf, shoulderRange[0], shoulderRange[1], 0.0, 1.0)
        theta4 =utils.changeRange(arm.yEndEf, elbowRange[0], elbowRange[1], 0.0, 1.0)
        arm_angles2 = np.array([theta3,theta4])
        
        
        
        if utils.distance(arm_angles, arm_angles2) <  3 * sigma:
            agvw2 = patches.Wedge(np.array([arm.xEndEf,arm.yEndEf]), 3* sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 0.07, lw = 1)
            ax1.add_artist(agvw2)
    
        if utils.distance(arm_angles, arm_angles2) <  2 * sigma:
            agvw1 = patches.Wedge(np.array([arm.xEndEf,arm.yEndEf]), 2* sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 0.1, lw = 1)
            ax1.add_artist(agvw1)
            
        if utils.distance(arm_angles, arm_angles2) <  sigma:
            agvw = patches.Wedge(np.array([arm.xEndEf,arm.yEndEf]), sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 1.0, lw = 1)
            ax1.add_artist(agvw)




