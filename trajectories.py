# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:51:14 2018

@author: Alex
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import utilities as utils


from arm import Arm
from games import armReaching6targets




shoulderRange = np.deg2rad(np.array([-30.0, 180.0]))
elbowRange    = np.deg2rad(np.array([  0.0, 180.0])) 



CEREBELLUM = True
INTRALAMINAR_NUCLEI = False

ATAXIA = True
damageMag = 10.0




actETA1 = 6.0 * 10 ** ( -1)
actETA2 = 1.0 * 10 ** (- 3)
critETA = 6.0 * 10 ** ( -6)
cbETA   = 6.0 * 10 ** ( -2)



goalRange = 0.03


seed = 0
maxEpoch = 1000
maxStep = 150
startPlotting= 950

mydir = os.getcwd



        
        
if CEREBELLUM == True:
    if INTRALAMINAR_NUCLEI == True:              
        os.chdir("C:\Users/Alex/Desktop/targets6/data/intralaminarNuclei/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s/" % (actETA1,actETA2,critETA,cbETA))
    else:
        os.chdir("C:\Users/Alex/Desktop/targets6/data/onlyCerebellum/actETA1=%s_critETA=%s_cbETA=%s/" % (actETA1,critETA,cbETA))
    
    if ATAXIA == True:
        os.chdir(os.curdir + "/cerebellumDamage=" + str(damageMag) + "/")
        
            
else:
    os.chdir("C:\Users/Alex/Desktop/targets6/data/onlyGanglia/actETA1=%s_critETA=%s/" % (actETA1,critETA))



gameGoalPos = np.zeros(2)


trajectories              = np.load("gameTrajectories_seed=%s.npy" %(seed))
armAngles                 = np.load("gameTrialArmAngles_seed=%s.npy" % (seed))
gangliaAngles             = np.load("gameTrialGangliaAngles_seed=%s.npy" % (seed))

if CEREBELLUM == True:
    cerebAngles           = np.load("gameTrialCerebAngles_seed=%s.npy" %(seed)) 

goalPosition              = np.load("goalPositionHistory_seed=%s.npy" %(seed))
goalAngles                = np.load("goalAnglesHistory_seed=%s.npy" %(seed))
#velocity                  = np.load("gameVelocity_seed=%s.npy" %(seed) )
#accelleration             = np.load("gameAccelleration_seed=%s.npy" %(seed))
jerk                      = np.load("gameJerk_seed=%s.npy" %(seed))


#linearityIndexData = np.zeros([game.maxTrial,maxEpoch])
##asimmetryIndexData = np.zeros([game.maxTrial,maxEpoch])
#smoothnessIndexData = np.zeros([game.maxTrial,maxEpoch])



if __name__ == "__main__":
    
    game = armReaching6targets()
    game.init(maxStep, maxEpoch)
    
    arm = Arm(shoulderRange, elbowRange)
  #  arm.__init__()
    
    
    for epoch in xrange(maxEpoch):
        
        if epoch == startPlotting:
                
                
            fig1   = plt.figure("Workspace", figsize=(16     ,16))
            gs = plt.GridSpec(8,8)
            
            text1 = plt.figtext(.02, .72, "epoch = %s" % (0), style='italic', bbox={'facecolor':'yellow'})
            text2 = plt.figtext(.02, .62, "trial = %s" % (0), style='italic', bbox={'facecolor':'lightblue'})
            
            text3 = plt.figtext(.30, .20, "linearity index = %s" % (0), style='italic', bbox={'facecolor':'lightblue'})
            
            
            
            ax1 = fig1.add_subplot(gs[0:8, 0:5])
            
         #   line1, = ax1.plot([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf], 'k-', color = 'pink', linewidth = 10)
      #      point, = ax1.plot([arm.xEndEf], [arm.yEndEf], 'o', color = 'black' , markersize= 20)
       
            
            
            ax1.set_xlim([-0.4,0.05])
            ax1.set_ylim([ 0.2,0.6])
            
            circle1 = plt.Circle((game.goalList[0]), goalRange, color = 'yellow') 
            edgecircle1 = plt.Circle((game.goalList[0]), goalRange, color = 'black', fill = False) 
            ax1.add_artist(circle1)
            ax1.add_artist(edgecircle1)
            
            circle2 = plt.Circle((game.goalList[1]), goalRange, color = 'yellow') 
            edgecircle2 = plt.Circle((game.goalList[1]), goalRange, color = 'black', fill = False) 
            ax1.add_artist(circle2)
            ax1.add_artist(edgecircle2)
            
            circle3 = plt.Circle((game.goalList[2]), goalRange, color = 'yellow') 
            edgecircle3 = plt.Circle((game.goalList[2]), goalRange, color = 'black', fill = False) 
            ax1.add_artist(circle3)
            ax1.add_artist(edgecircle3)
            
            circle4 = plt.Circle((game.goalList[3]), goalRange, color = 'yellow') 
            edgecircle4 = plt.Circle((game.goalList[3]), goalRange, color = 'black', fill = False) 
            ax1.add_artist(circle4)
            ax1.add_artist(edgecircle4)
            
            circle5 = plt.Circle((game.goalList[4]), goalRange, color = 'yellow') 
            edgecircle5 = plt.Circle((game.goalList[4]), goalRange, color = 'black', fill = False) 
            ax1.add_artist(circle5)
            ax1.add_artist(edgecircle5)
            
            circle6 = plt.Circle((game.goalList[5]), goalRange, color = 'yellow') 
            edgecircle6 = plt.Circle((game.goalList[5]), goalRange, color = 'black', fill = False) 
            ax1.add_artist(circle6)
            ax1.add_artist(edgecircle6)
            
            rewardCircle = plt.Circle((game.goalList[0]), goalRange, color = 'red')  
            ax1.add_artist(rewardCircle)
            
            
            
            
            ax3 =  fig1.add_subplot(gs[0:1, 6:8])
            ax3.set_ylim([0,1.5])
            ax3.set_xlim([0,maxStep])
            ax3.set_yticks(np.arange(0, 1.5, 0.5))
            ax3.set_xticks(np.arange(0, maxStep, 20))
            title3 = plt.figtext(.79, 0.90, "VELOCITY" , style='normal', bbox={'facecolor':'orangered'})
            text5 = plt.figtext(.74, .75, "asimmetry index = %s" % (0), style='italic', bbox={'facecolor':'lightblue'})
            
            
            
        #    ax4 =  fig1.add_subplot(gs[2:3, 6:8])
         #   ax4.set_ylim([0,4])
        #    ax4.set_xlim([0,100])
       #     ax4.set_xticks(np.arange(0, 100, 10))
        #    title4 = plt.figtext(.767, .70, "ACCELLERATION" , style='normal', bbox={'facecolor':'orangered'})
      
            
            ax5 =  fig1.add_subplot(gs[2:3, 6:8])
            ax5.set_ylim([-2000,2000])
            ax5.set_xlim([0,maxStep])
            ax5.set_yticks(np.arange(-2000, 2000, 1000))
            ax5.set_xticks(np.arange(0, maxStep, 20))
            
            title5 = plt.figtext(.8, .70, "JERK" , style='normal', bbox={'facecolor':'orangered'})
            text4 = plt.figtext(.72, .55, "smoothness index = %s" % (0), style='italic', bbox={'facecolor':'lightblue'})
            
            ax6 =fig1.add_subplot(gs[5:6, 6:8])
            ax6.set_xlim([0,100])
            ax6.set_xticks(np.arange(0, maxStep, 20))
            ax6.set_ylim([-1.0, np.pi])
            
            title6 = plt.figtext(.76, .41, "SHOULDER ANGLE" , style='normal', bbox={'facecolor':'orangered'})
            
            ax7 =fig1.add_subplot(gs[7:8, 6:8])
            ax7.set_xlim([0,100])
            ax7.set_xticks(np.arange(0, maxStep, 20))
            ax7.set_ylim([0.0, np.pi])
            
            title7 = plt.figtext(.77, .21, "ELBOW ANGLE" , style='normal', bbox={'facecolor':'orangered'})
            
       
            
            
            
            
        
        
        if epoch >= startPlotting:                
            text1.set_text("epoch = %s" % (epoch +1))
        
            
        for trial in xrange(game.maxTrial):
            
            print trial, epoch
            
      #      asimmetryIndex = 0
            
          #  print trial
            

            gameGoalPos = goalPosition[:,trial,epoch].copy()
            
            
            trialArmAngles = armAngles[:,:,trial,epoch].copy()
            trialGoalAngles = goalAngles[:,:,trial,epoch].copy()
            trialGangliaAngles = gangliaAngles[:,:,trial,epoch].copy()
            
            if CEREBELLUM == True:
                trialCerebAngles = cerebAngles[:,:,trial,epoch].copy()
            
            
            
            
            trialTraj = trajectories[:,:,trial,epoch].copy()
            
            if trialTraj[0,:].any() != 0 :
                trimmedTraj = utils.trimTraj(trialTraj) 
            
                minDistance = utils.distance(trimmedTraj[:,0], trimmedTraj[:,len(trimmedTraj[0,:]) -1])
                trajLen = utils.trajLen(trimmedTraj)
            
            
                trialTangVel = np.zeros(len(trimmedTraj[0,:]))
            

                for step in xrange(len(trimmedTraj[0,:])):
                
                    if step > 0:
               #     print trimmedTraj[:,step]
          #          print utils.distance(trimmedTraj[:,step],trimmedTraj[:,step -1]) / arm.dt
                        trialTangVel[step] = utils.distance(trimmedTraj[:,step],trimmedTraj[:,step -1]) / arm.dt
            
            #   trialVelocity = velocity[:,trial,epoch].copy()
         #   trialAccelleration = accelleration[:,trial,epoch].copy()
                trialTangJerk = np.trim_zeros(jerk[:,trial,epoch], 'b')
            

            
           
            
            
            
            
            
            
                trialdXdY = np.zeros([ 2, len(trimmedTraj[0,:]) ])
            
                trialdXdY[0,:] = np.ediff1d(trimmedTraj[0,:], to_begin = np.array([0]))  / arm.dt
                trialdXdY[1,:] = np.ediff1d(trimmedTraj[1,:], to_begin = np.array([0]))  / arm.dt
            
           
                trialddXddY = np.zeros([ 2, len(trimmedTraj[0,:]) ])
            
                trialddXddY[0,:] = np.ediff1d(trialdXdY[0,:], to_begin = np.array([0]))  / arm.dt
                trialddXddY[1,:] = np.ediff1d(trialdXdY[1,:], to_begin = np.array([0]))  / arm.dt
                       
            
                       
                       
                trialdddXdddY = np.zeros([ 2, len(trimmedTraj[0,:]) ])
            
                trialdddXdddY[0,:] = np.ediff1d(trialddXddY[0,:], to_begin = np.array([0]))  / arm.dt
                trialdddXdddY[1,:] = np.ediff1d(trialddXddY[1,:], to_begin = np.array([0]))  / arm.dt
                    
                
                trialJerk = np.mean(trialdddXdddY)
            
            
            
            
            
            
            
            
            
            
                linearityIndex = (utils.linearityIndex(trajLen, minDistance) -1.) * 100
            
            
            
            
           
            
          #  if epoch > startPlotting:
          #      print minDistance ,trial
          #  trajLen = utils.trajLen(trialTraj)
    #        minDistance = utils.distance(trialTraj[:,0],trialTraj[:,0])
 #           linearityIndex = utils.linearityIndex(trialTraj, gameGoalPos, goalRange)
#
                smoothnessIndex = utils.smoothnessIndex(trialdddXdddY, trajLen, arm.dt)
            
            
                asimmetryIndex = utils.asimmetryIndex(trialTangVel)
            

                
            
            
            if epoch >= startPlotting:
                    
                text2.set_text("trial = %s" % (trial +1))
                text3.set_text("linearity index = %s" % (linearityIndex))
                text4.set_text("smoothness index = %s" % (smoothnessIndex))
                text5.set_text("asimmetry index = %s" % (asimmetryIndex))
                
                ax1.cla()
                ax1.set_xlim([-0.4,0.05])
                ax1.set_ylim([ 0.2,0.6])
                
                circle1 = plt.Circle((game.goalList[0]), goalRange, color = 'yellow') 
                edgecircle1 = plt.Circle((game.goalList[0]), goalRange, color = 'black', fill = False) 
                ax1.add_artist(circle1)
                ax1.add_artist(edgecircle1)
                
                circle2 = plt.Circle((game.goalList[1]), goalRange, color = 'yellow') 
                edgecircle2 = plt.Circle((game.goalList[1]), goalRange, color = 'black', fill = False) 
                ax1.add_artist(circle2)
                ax1.add_artist(edgecircle2)
                
                circle3 = plt.Circle((game.goalList[2]), goalRange, color = 'yellow') 
                edgecircle3 = plt.Circle((game.goalList[2]), goalRange, color = 'black', fill = False) 
                ax1.add_artist(circle3)
                ax1.add_artist(edgecircle3)
                
                circle4 = plt.Circle((game.goalList[3]), goalRange, color = 'yellow') 
                edgecircle4 = plt.Circle((game.goalList[3]), goalRange, color = 'black', fill = False) 
                ax1.add_artist(circle4)
                ax1.add_artist(edgecircle4)
                
                circle5 = plt.Circle((game.goalList[4]), goalRange, color = 'yellow') 
                edgecircle5 = plt.Circle((game.goalList[4]), goalRange, color = 'black', fill = False) 
                ax1.add_artist(circle5)
                ax1.add_artist(edgecircle5)
                
                circle6 = plt.Circle((game.goalList[5]), goalRange, color = 'yellow') 
                edgecircle6 = plt.Circle((game.goalList[5]), goalRange, color = 'black', fill = False) 
                ax1.add_artist(circle6)
                ax1.add_artist(edgecircle6)
                
                rewardCircle = plt.Circle((gameGoalPos), goalRange, color = 'red')  
                ax1.add_artist(rewardCircle)
        
        
                traj = plt.Line2D(trimmedTraj[0] , trimmedTraj[1], color = 'red')
                ax1.add_artist(traj)
                
            
                   
                linearVelocityPlot, = ax3.plot(trialTangVel, color='blue')
             #   linearAccellerationyPlot, = ax4.plot(np.trim_zeros(trialAccelleration, 'b'), color='blue')
                linearJerkPlot, = ax5.plot(trialTangJerk, color='blue')
                
                
                
                desiredShoulderAngle, = ax6.plot(np.trim_zeros(trialGoalAngles[0], 'b'), color='red')
                shoulderGangliaAngle, = ax6.plot(np.trim_zeros(trialGangliaAngles[0], 'b'), color='blue')
                currShoulderAngle, = ax6.plot(np.trim_zeros(trialArmAngles[0], 'b'), color='black')
                
                desiredElbowAngle, = ax7.plot(np.trim_zeros(trialGoalAngles[1], 'b'), color='red')
                elbowGangliaAngle = ax7.plot(np.trim_zeros(trialGangliaAngles[1], 'b'), color='blue')
                currElbowAngle, = ax7.plot(np.trim_zeros(trialArmAngles[1], 'b'), color='black')
                
                if CEREBELLUM == True:
                    shoulderCerebAngle, = ax6.plot(np.trim_zeros(trialCerebAngles[0]), color='orange')
                    elbowCerebAngle, = ax7.plot(np.trim_zeros(trialCerebAngles[1]), color='orange')
                

                    

                plt.pause(3.0)
                    
                    
                ax3.cla()
                ax3.set_ylim([0,1.5])
                ax3.set_xlim([0,maxStep])
                ax3.set_yticks(np.arange(0, 1.5, 0.5))
                ax3.set_xticks(np.arange(0, maxStep, 20))
                
             #   ax4.cla()
            #    ax4.set_ylim([0,4])
            #    ax4.set_xlim([0,100])
              #  ax4.set_yticks(np.arange(0, 4, 0.8))
           #     ax4.set_xticks(np.arange(0, 100, 10))
                
                ax5.cla()
                ax5.set_ylim([-2000,2000])
                ax5.set_xlim([0,maxStep])
                ax5.set_yticks(np.arange(-2000, 2000, 1000))
                ax5.set_xticks(np.arange(0, maxStep, 20))
                
                ax6.cla()
                ax6.set_xlim([0,maxStep])
                ax6.set_xticks(np.arange(0, maxStep, 20))
                ax6.set_ylim([-1.0, np.pi])
                
                ax7.cla()
                ax7.set_xlim([0,maxStep])
                ax7.set_xticks(np.arange(0, maxStep, 20))
                ax7.set_ylim([0.0, np.pi])
                
                
                                  
                                  
                                  
                                  

                
                

        
        
    
    
    