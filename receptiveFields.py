# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 00:07:49 2018

@author: Alex
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.mlab import bivariate_normal
from mpl_toolkits.mplot3d import Axes3D

import utilities as utils
from arm import Arm





xVisionRange = np.array([-0.6, 0.4]) 
yVisionRange = np.array([ 0.0, 1.0])  





shoulderRange = np.deg2rad(np.array([-60.0, 150.0]))
elbowRange    = np.deg2rad(np.array([  0.0, 180.0])) 

arm = Arm(shoulderRange, elbowRange)






goalRange = 0.03

goalList = [np.array([-0.07 -.05, 0.16 +0.04]),
            np.array([ 0.10 -.05, 0.16 +0.04]),
            np.array([-0.24 -.05, 0.16 +0.04]),
            np.array([-0.07 -.05, 0.42 +0.04]),        
            np.array([ 0.10 -.05, 0.42 +0.04]),
            np.array([-0.24 -.05, 0.42 +0.04])]


normalized_goal0_position = np.array([utils.changeRange(goalList[0][0], xVisionRange[0], xVisionRange[1], 0.0, 1.0), utils.changeRange(goalList[0][1], yVisionRange[0], yVisionRange[1], 0.0, 1.0) ])
normalized_goal1_position = np.array([utils.changeRange(goalList[1][0], xVisionRange[0], xVisionRange[1], 0.0, 1.0), utils.changeRange(goalList[1][1], yVisionRange[0], yVisionRange[1], 0.0, 1.0) ])
normalized_endeff_pos = np.array([utils.changeRange(arm.xEndEf, xVisionRange[0], xVisionRange[1], 0.0, 1.0), utils.changeRange(arm.yEndEf, yVisionRange[0], yVisionRange[1], 0.0, 1.0) ])




x_N = 41
y_N = 41

sigma = 1.0 / ( 2 * (x_N -1))

x_grid = np.linspace(xVisionRange[0],xVisionRange[1],x_N)
y_grid = np.linspace(yVisionRange[0],yVisionRange[1],y_N)
rawGrid = np.meshgrid(x_grid,y_grid)
grid = np.vstack(([rawGrid[0].T],[rawGrid[1].T])).T
                
goal_state = np.zeros(x_N*y_N)
goal_state1 = np.zeros(x_N*y_N)
endeff_state = np.zeros(x_N*y_N)
endeff_state1 = np.zeros(x_N*y_N)

#X,Y = np.meshgrid(utils.changeRange(x_grid, xVisionRange[0],xVisionRange[1], 0.0, 1.0) ,utils.changeRange(y_grid, yVisionRange[0],yVisionRange[1], 0.0, 1.0)) 
                
#normalized_raw_grid = np.meshgrid(utils.changeRange(x_grid, xVisionRange[0],xVisionRange[1], 0.0, 1.0) ,utils.changeRange(y_grid, yVisionRange[0],yVisionRange[1], 0.0, 1.0)) 
#normalized_grid = np.vstack(([normalized_raw_grid[0].T],[normalized_raw_grid[1].T])).T

goal_vision_grid = utils.buildGrid(0.0,1.0,x_N,0.0,1.0,y_N).reshape(2,x_N*y_N)
                
                
                
                


fig1   = plt.figure("Goal vision receptive field", figsize=(9,8))
ax1    = fig1.add_subplot(121, aspect='equal')
ax1.set_xlim([-1.0,1.0])
ax1.set_ylim([-1.0,1.0])

#ax3    = fig1.add_subplot(222, aspect='equal')
#ax3.set_xlim([-0.7,0.5])
#ax3.set_ylim([-0.2,0.8])


ax5    = fig1.add_subplot(122, projection ='3d', aspect='equal')
ax5.set_xlim([ 0.0, 1.0])
ax5.set_ylim([ 0.0, 1.0])
ax5.set_zlim([ 0.0, 1.0])
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel(r'$\chi$')



goal_state = utils.rbf2D(normalized_goal0_position, goal_state,goal_vision_grid,sigma)
goal_activation = np.dstack([goal_vision_grid[1], goal_vision_grid[0], goal_state])
goal_activation = goal_activation[0]

activation = ax5.plot_trisurf(goal_activation[:,1], goal_activation[:,0], goal_activation[:,2], cmap='copper',linewidth=0)
fig1.colorbar(activation)
#ax7    = fig1.add_subplot(224, projection ='3d', aspect='equal')
#ax7.set_xlim([ 0.0, 1.0])
#ax7.set_ylim([ 0.0, 1.0])
#ax7.set_zlim([ 0.0, 1.0])
#ax7.set_xlabel('X')
#ax7.set_ylabel('Y')
#ax7.set_zlabel(r'$\chi$')


#goal_state1 = utils.rbf2D(normalized_goal1_position, goal_state1,goal_vision_grid,sigma)
#goal_activation1 = np.dstack([goal_vision_grid[0], goal_vision_grid[1], goal_state1])
#goal_activation1 = goal_activation1[0]

#activation1 = ax7.plot_trisurf(goal_activation1[:,1], goal_activation1[:,0], goal_activation1[:,2], cmap='viridis',linewidth=0)



fig2   = plt.figure("End-effector vision receptive field", figsize=(9,8))
ax2    = fig2.add_subplot(121, aspect='equal')
ax2.set_xlim([-1.0,1.0])
ax2.set_ylim([-1.0,1.0])

#ax4    = fig2.add_subplot(222, aspect='equal')
#ax4.set_xlim([-0.7,0.5])
#ax4.set_ylim([-0.2,0.8])

ax6    = fig2.add_subplot(122, projection ='3d', aspect='equal')
ax6.set_xlim([-0.0,1.0])
ax6.set_ylim([-0.0,1.0])
ax6.set_zlim([-0.0,1.0])

endeff_state = utils.rbf2D(normalized_endeff_pos, endeff_state,goal_vision_grid,sigma)
endeff_activation = np.dstack([goal_vision_grid[1], goal_vision_grid[0], endeff_state])
endeff_activation = endeff_activation[0]

activation2 = ax6.plot_trisurf(endeff_activation[:,1], endeff_activation[:,0], endeff_activation[:,2], cmap='copper',linewidth=0)
fig2.colorbar(activation2)
#ax8    = fig2.add_subplot(224, projection ='3d', aspect='equal')
#ax8.set_xlim([-0.0,1.0])
#ax8.set_ylim([-0.0,1.0])
#ax8.set_zlim([-0.0,1.0])











#fig3   = plt.figure("End-effector proprioception receptive field", figsize=(9,8))
#ax9    = fig3.add_subplot(221, aspect='equal')
#ax9.set_xlim([-1.0,1.0])
#ax9.set_ylim([-1,0,1.0])




#goal_vision_limit = patches.Rectangle(np.array([xVisionRange[0], yVisionRange[0]]), np.sqrt((xVisionRange[0] - xVisionRange[1])**2), np.sqrt((yVisionRange[0] - yVisionRange[1])**2) ,linewidth=1,edgecolor='black',facecolor='lightgrey')
#endEff_vision_limit = patches.Rectangle(np.array([xVisionRange[0], yVisionRange[0]]), np.sqrt((xVisionRange[0] - xVisionRange[1])**2), np.sqrt((yVisionRange[0] - yVisionRange[1])**2) ,linewidth=1,edgecolor='black',facecolor='lightgrey')

#ax1.add_patch(goal_vision_limit)
#ax3.add_patch(goal_vision_limit)

#x2.add_patch(endEff_vision_limit)
#ax4.add_patch(endEff_vision_limit)







for i in xrange(len(grid[0])):
    for j in xrange(len(grid[1])):        
        
        goal_3sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 3 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.3, lw = 1)
        goal_2sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 2 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.6, lw = 1)        
        goal_1sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 1 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 1.0, lw = 1)
    
        ax1.add_artist(goal_3sigma)
        ax1.add_artist(goal_2sigma)
        ax1.add_artist(goal_1sigma)
        
        endEff_3sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 3 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.3, lw = 1)
        endEff_2sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 2 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.6, lw = 1)        
        endEff_1sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 1 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 1.0, lw = 1)
        
        ax2.add_artist(endEff_3sigma)
        ax2.add_artist(endEff_2sigma)
        ax2.add_artist(endEff_1sigma)
        
        
for i in xrange(len(grid[0])):
    for j in xrange(len(grid[1])):  
        
        goal2_3sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 3 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.3, lw = 1)
        goal2_2sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 2 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.6, lw = 1)        
        goal2_1sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 1 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 1.0, lw = 1)
    
    #    ax3.add_artist(goal2_3sigma)
    #    ax3.add_artist(goal2_2sigma)
    #    ax3.add_artist(goal2_1sigma)
        
        
        endEff2_3sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 3 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.3, lw = 1)
        endEff2_2sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 2 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.6, lw = 1)        
        endEff2_1sigma = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 1 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 1.0, lw = 1)
        
    #    ax4.add_artist(endEff2_3sigma)
    #    ax4.add_artist(endEff2_2sigma)
    #    ax4.add_artist(endEff2_1sigma)



for i in xrange(len(grid[0])):
    for j in xrange(len(grid[1])):
        
        goal2_3sigma = patches.Wedge(np.array(utils.changeRange(grid[i,j,:][0], 0.0,1.0,shoulderRange[0],shoulderRange[1]),utils.changeRange(grid[i,j,:][0], 0.0,1.0,elbowRange[0],elbowRange[1])), 3 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.3, lw = 1)
        goal2_2sigma = patches.Wedge(np.array(utils.changeRange(grid[i,j,:][0], 0.0,1.0,shoulderRange[0],shoulderRange[1]),utils.changeRange(grid[i,j,:][0], 0.0,1.0,elbowRange[0],elbowRange[1])), 2 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 0.6, lw = 1)        
        goal2_1sigma = patches.Wedge(np.array(utils.changeRange(grid[i,j,:][0], 0.0,1.0,shoulderRange[0],shoulderRange[1]),utils.changeRange(grid[i,j,:][0], 0.0,1.0,elbowRange[0],elbowRange[1])), 1 *sigma, 0.0,360, fc= 'lightblue', ec='black' , alpha = 1.0, lw = 1)
    
  #      ax9.add_artist(goal2_3sigma)
  #      ax9.add_artist(goal2_2sigma)
  #      ax9.add_artist(goal2_1sigma)
        


#Z = bivariate_normal(X,Y,sigma,sigma,grid[i,j,:][0],grid[i,j,:][1])
#ax5.plot_surface(X, Y, Z, cmap='viridis',linewidth=1)


        
for i in xrange(len(grid[0])):
    for j in xrange(len(grid[1])):
        
        
        
        if utils.distance(goalList[0], np.array([grid[i,j,:][0],grid[i,j,:][1]])) <  3 * sigma:
            agvw2 = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 3* sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 0.07, lw = 1)
            ax1.add_artist(agvw2)
    
        if utils.distance(goalList[0], np.array([grid[i,j,:][0],grid[i,j,:][1]])) <  2 * sigma:
            agvw1 = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 2* sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 0.1, lw = 1)
            ax1.add_artist(agvw1)
            
        if utils.distance(goalList[0], np.array([grid[i,j,:][0],grid[i,j,:][1]])) <  sigma:
            agvw = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 1.0, lw = 1)
            ax1.add_artist(agvw)
        
     #   Z = bivariate_normal(X,Y,sigma,sigma,grid[i,j,:][0],grid[i,j,:][1])
        
        
        
        #  ax5.plot_surface(X, Y, Z, cmap='viridis',linewidth=0)
        
        
        
   #     if utils.distance(goalList[5], np.array([grid[i,j,:][0],grid[i,j,:][1]])) <  3 * sigma:
   #         agv2w2 = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 3* sigma, 0.0,360, fc= 'lightgreen', ec='black' , alpha = 0.07, lw = 1)
   #         ax3.add_artist(agv2w2)
    
   #     if utils.distance(goalList[5], np.array([grid[i,j,:][0],grid[i,j,:][1]])) <  2 * sigma:
   #         agv2w1 = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 2* sigma, 0.0,360, fc= 'lightgreen', ec='black' , alpha = 0.1, lw = 1)
   #         ax3.add_artist(agv2w1)
            
   #     if utils.distance(goalList[5], np.array([grid[i,j,:][0],grid[i,j,:][1]])) <  sigma:
   #         agv2w = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), sigma, 0.0,360, fc= 'lightgreen', ec='black' , alpha = 1.0, lw = 1)
   #         ax3.add_artist(agv2w)
            
        
        
        
        if utils.distance(np.array([arm.xEndEf,arm.yEndEf]), np.array([grid[i,j,:][0],grid[i,j,:][1]]))  <  3 * sigma:
            aefv2w2 = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 3* sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 0.05, lw = 1)
            ax2.add_artist(aefv2w2)
    
        if utils.distance(np.array([arm.xEndEf,arm.yEndEf]), np.array([grid[i,j,:][0],grid[i,j,:][1]])) <  2 * sigma:
            aefv2w1= patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 2* sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 0.1, lw = 1)
            ax2.add_artist(aefv2w1)
            
        if utils.distance(np.array([arm.xEndEf,arm.yEndEf]), np.array([grid[i,j,:][0],grid[i,j,:][1]])) <  sigma:
            aefv2w = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 1.0, lw = 1)
            ax2.add_artist(aefv2w)
            

 

           

            
            




























            
            


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

rewardCircle = plt.Circle((goalList[0]), goalRange, color = 'red', alpha = 0.5)  
ax1.add_artist(rewardCircle)

arm1, = ax1.plot([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf], 'k-', color = 'pink', linewidth = 10)
hand1, = ax1.plot([arm.xEndEf], [arm.yEndEf], 'o', color = 'black' , markersize= 20, alpha = 0.3)





circle1 = plt.Circle((goalList[0]), goalRange, color = 'yellow') 
edgecircle1 = plt.Circle((goalList[0]), goalRange, color = 'black', fill = False) 
ax2.add_artist(circle1)
ax2.add_artist(edgecircle1)

circle2 = plt.Circle((goalList[1]), goalRange, color = 'yellow') 
edgecircle2 = plt.Circle((goalList[1]), goalRange, color = 'black', fill = False) 
ax2.add_artist(circle2)
ax2.add_artist(edgecircle2)

circle3 = plt.Circle((goalList[2]), goalRange, color = 'yellow') 
edgecircle3 = plt.Circle((goalList[2]), goalRange, color = 'black', fill = False) 
ax2.add_artist(circle3)
ax2.add_artist(edgecircle3)

circle4 = plt.Circle((goalList[3]), goalRange, color = 'yellow') 
edgecircle4 = plt.Circle((goalList[3]), goalRange, color = 'black', fill = False) 
ax2.add_artist(circle4)
ax2.add_artist(edgecircle4)

circle5 = plt.Circle((goalList[4]), goalRange, color = 'yellow') 
edgecircle5 = plt.Circle((goalList[4]), goalRange, color = 'black', fill = False) 
ax2.add_artist(circle5)
ax2.add_artist(edgecircle5)

circle6 = plt.Circle((goalList[5]), goalRange, color = 'yellow') 
edgecircle6 = plt.Circle((goalList[5]), goalRange, color = 'black', fill = False) 
ax2.add_artist(circle6)
ax2.add_artist(edgecircle6)

arm2, = ax2.plot([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf], 'k-', color = 'pink', linewidth = 10)
hand2, = ax2.plot([arm.xEndEf], [arm.yEndEf], 'o', color = 'black' , markersize= 20 , alpha = 0.3)







#arm.theta1 = np.deg2rad(60.0)
#arm.theta2 = np.deg2rad(120.0)
#arm.getElbowPose()
#arm.getEndEffectorPose()

#normalized_endeff_pos1 = np.array([utils.changeRange(arm.xEndEf, xVisionRange[0], xVisionRange[1], 0.0, 1.0), utils.changeRange(arm.yEndEf, yVisionRange[0], yVisionRange[1], 0.0, 1.0) ])
#endeff_state1 = utils.rbf2D(normalized_endeff_pos1, endeff_state1,goal_vision_grid,sigma)
#endeff_activation1 = np.dstack([goal_vision_grid[1], goal_vision_grid[0], endeff_state1])
#endeff_activation1 = endeff_activation1[0]

#activation3 = ax8.plot_trisurf(endeff_activation1[:,1], endeff_activation1[:,0], endeff_activation1[:,2], cmap='viridis',linewidth=0)


"""
circle1 = plt.Circle((goalList[0]), goalRange, color = 'yellow') 
edgecircle1 = plt.Circle((goalList[0]), goalRange, color = 'black', fill = False) 
ax3.add_artist(circle1)
ax3.add_artist(edgecircle1)

circle2 = plt.Circle((goalList[1]), goalRange, color = 'yellow') 
edgecircle2 = plt.Circle((goalList[1]), goalRange, color = 'black', fill = False) 
ax3.add_artist(circle2)
ax3.add_artist(edgecircle2)

circle3 = plt.Circle((goalList[2]), goalRange, color = 'yellow') 
edgecircle3 = plt.Circle((goalList[2]), goalRange, color = 'black', fill = False) 
ax3.add_artist(circle3)
ax3.add_artist(edgecircle3)

circle4 = plt.Circle((goalList[3]), goalRange, color = 'yellow') 
edgecircle4 = plt.Circle((goalList[3]), goalRange, color = 'black', fill = False) 
ax3.add_artist(circle4)
ax3.add_artist(edgecircle4)

circle5 = plt.Circle((goalList[4]), goalRange, color = 'yellow') 
edgecircle5 = plt.Circle((goalList[4]), goalRange, color = 'black', fill = False) 
ax3.add_artist(circle5)
ax3.add_artist(edgecircle5)

circle6 = plt.Circle((goalList[5]), goalRange, color = 'yellow') 
edgecircle6 = plt.Circle((goalList[5]), goalRange, color = 'black', fill = False) 
ax3.add_artist(circle6)
ax3.add_artist(edgecircle6)

rewardCircle = plt.Circle((goalList[5]), goalRange, color = 'red')  
ax3.add_artist(rewardCircle)

arm3, = ax3.plot([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf], 'k-', color = 'pink', linewidth = 10)
hand3, = ax3.plot([arm.xEndEf], [arm.yEndEf], 'o', color = 'black' , markersize= 20)



circle1 = plt.Circle((goalList[0]), goalRange, color = 'yellow') 
edgecircle1 = plt.Circle((goalList[0]), goalRange, color = 'black', fill = False) 
ax4.add_artist(circle1)
ax4.add_artist(edgecircle1)

circle2 = plt.Circle((goalList[1]), goalRange, color = 'yellow') 
edgecircle2 = plt.Circle((goalList[1]), goalRange, color = 'black', fill = False) 
ax4.add_artist(circle2)
ax4.add_artist(edgecircle2)

circle3 = plt.Circle((goalList[2]), goalRange, color = 'yellow') 
edgecircle3 = plt.Circle((goalList[2]), goalRange, color = 'black', fill = False) 
ax4.add_artist(circle3)
ax4.add_artist(edgecircle3)

circle4 = plt.Circle((goalList[3]), goalRange, color = 'yellow') 
edgecircle4 = plt.Circle((goalList[3]), goalRange, color = 'black', fill = False) 
ax4.add_artist(circle4)
ax4.add_artist(edgecircle4)

circle5 = plt.Circle((goalList[4]), goalRange, color = 'yellow') 
edgecircle5 = plt.Circle((goalList[4]), goalRange, color = 'black', fill = False) 
ax4.add_artist(circle5)
ax4.add_artist(edgecircle5)

circle6 = plt.Circle((goalList[5]), goalRange, color = 'yellow') 
edgecircle6 = plt.Circle((goalList[5]), goalRange, color = 'black', fill = False) 
ax4.add_artist(circle6)
ax4.add_artist(edgecircle6)




for i in xrange(len(grid[0])):
    for j in xrange(len(grid[1])):
        
        if utils.distance(np.array([arm.xEndEf,arm.yEndEf]), np.array([grid[i,j,:][0],grid[i,j,:][1]]))  <  3 * sigma:
            aef2vw2 = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 3* sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 0.05, lw = 1)
            ax4.add_artist(aef2vw2)
    
        if utils.distance(np.array([arm.xEndEf,arm.yEndEf]), np.array([grid[i,j,:][0],grid[i,j,:][1]])) <  2 * sigma:
            aef2vw1= patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), 2* sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 0.1, lw = 1)
            ax4.add_artist(aef2vw1)
            
        if utils.distance(np.array([arm.xEndEf,arm.yEndEf]), np.array([grid[i,j,:][0],grid[i,j,:][1]])) <  sigma:
            aef2vw = patches.Wedge(np.array([grid[i,j,:][0],grid[i,j,:][1]]), sigma, 0.0,360, fc= 'orange', ec='black' , alpha = 1.0, lw = 1)
            ax4.add_artist(aef2vw)
        


arm4, = ax4.plot([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf], 'k-', color = 'pink', linewidth = 10)
hand4, = ax4.plot([arm.xEndEf], [arm.yEndEf], 'o', color = 'black' , markersize= 20)

"""
