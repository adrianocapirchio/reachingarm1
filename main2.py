# -*- coding: utf-8 -*-
"""
Created on Tue May 08 18:19:18 2018

@author: Alex
"""

from arm import Arm
from basalGanglia import actorCritic
from cerebellum import Cerebellum 
from motor_cortex import Motor_cortex
from games import armReaching6targets


import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import utilities as utils
import os



shoulderRange = np.deg2rad(np.array([-60.0, 150.0]))
elbowRange    = np.deg2rad(np.array([  0.0, 180.0])) 


xVisionRange = np.array([-0.6, 0.4]) 
yVisionRange = np.array([ 0.0, 1.0])  
netGoalVision = np.zeros(2)
netAgentVision = np.zeros(2)







TRAINING = True
GANGLIA_NOISE = True
gangliaTime = 1
epsi = 0.0
perfBuff = 400

MULTINET = False




CEREBELLUM = True
MOTOR_CORTEX = False
INTRALAMINAR_NUCLEI = False




CEREBELLUM_DAMAGE = False
damageMag = 1.2
TdCS = False
TdCSMag = 1.2


cerebellumTime = 1




Kp = 5.0
Kd = 0.5





VISION = True
GOAL_VISION = True
AGENT_VISION = True

PROPRIOCEPTION = True
SHOULDER_PROPRIOCEPTION = True
ELBOW_PROPRIOCEPTION = True




avgStats = 10



loadData = False
saveData = True










goalRange = 0.03
maxSeed = 1
maxEpoch = 1500
maxStep = 150

startPlotting = 1500
startAtaxia = 1460
startTdCS = 1480










if __name__ == "__main__":
    
    
    for seed in xrange(maxSeed):
        
        
        
        
        CEREBELLUM_DAMAGE = False
        TdCS == False
        
        ataxiaNoise = 0
        
        
        game = armReaching6targets()
        game.init(maxStep, maxEpoch)
    
        arm = Arm(shoulderRange, elbowRange)
        #   arm.__init__()
    
        bg = actorCritic()
        bg.init(MULTINET, VISION, GOAL_VISION , AGENT_VISION, \
                PROPRIOCEPTION, ELBOW_PROPRIOCEPTION, SHOULDER_PROPRIOCEPTION,\
                maxStep, game.maxTrial, \
                game.goalList, perfBuff, \
                cerebellumTime, DOF = 2)
    

    
        if CEREBELLUM == True:
            cb = Cerebellum()
            cb.init(MULTINET, game.goalList, bg.currState, maxStep)
            
            
        if MOTOR_CORTEX == True:
            mt = Motor_cortex()
            mt.init(bg.goalVisionState)



        avg5EpochAcc = np.zeros(avgStats)
        avg5EpochTime = np.ones(avgStats) * maxStep
        finalAvgAcc = np.zeros(maxEpoch/avgStats)
        finalAvgTime = np.ones(maxEpoch/avgStats) * maxStep
        epochAccurancy = np.zeros(game.maxTrial)
        epochGoalTime = np.ones(game.maxTrial) * maxStep 
        epochAvgAcc = np.zeros(1)
        epochAvgTime = np.zeros(1)
        
        
        
        if loadData == True:
            mydir = os.getcwd
            
            if CEREBELLUM == True:
                if INTRALAMINAR_NUCLEI == True:              
                    os.chdir("C:\Users/Alex/Desktop/targets6/data/intralaminarNuclei/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,cb.cbETA))
                    
                    bg.actW = np.load("actorWeights_seed=%s.npy" % (seed))
                    bg.critW = np.load("criticWeights_seed=%s.npy" % (seed))
                    
                else:
                    os.chdir("C:\Users/Alex/Desktop/targets6/data/onlyCerebellum/actETA1=%s_critETA=%s_cbETA=%s/" % (bg.ACT_ETA1,bg.CRIT_ETA,cb.cbETA))
                
                    cb.w = np.load("cerebellumWeights_seed=%s.npy" % (seed))
                    bg.actW = np.load("actorWeights_seed=%s.npy" % (seed))
                    bg.critW = np.load("criticWeights_seed=%s.npy" % (seed))
                
            else:
                os.chdir("C:\Users/Alex/Desktop/targets6/data/onlyGanglia/actETA1=%s_critETA=%s/" % (bg.ACT_ETA1,bg.CRIT_ETA))
                bg.actW = np.load("actorWeights_seed=%s.npy" % (seed))
                bg.critW = np.load("criticWeights_seed=%s.npy" % (seed))
                
                
                
                
        np.random.seed(seed)
        
        
        
        for epoch in xrange(maxEpoch):
            
            
            
            if CEREBELLUM == True:
                if epoch >= startAtaxia:
                    CEREBELLUM_DAMAGE = True
                    
            if CEREBELLUM_DAMAGE == True:
                if epoch > startTdCS:
                    TdCS == True
            
                    
            time = 0
            
            
            
            bg.epochReset(maxStep, cerebellumTime)
            arm.epochReset(shoulderRange, elbowRange)
            game.epochReset()
            game.currPos = np.array([arm.xEndEf,arm.yEndEf])  
            
            if MOTOR_CORTEX == True:
                mt.epochReset()
            
            
            if CEREBELLUM == True:
                cb.epochReset(maxStep)
            
            
            netOut = np.ones(2) * 0.5
            
            
            
            
            
            desiredAngles = np.ones(2) * 0.5
            desiredAngles[0] = utils.changeRange(desiredAngles[0], 0.,1.,shoulderRange[0],shoulderRange[1])
            desiredAngles[1] = utils.changeRange(desiredAngles[1], 0.,1., elbowRange[0],elbowRange[1])
            prvDesAngles = desiredAngles.copy()
            
            xDesAng = arm.L1*np.cos(desiredAngles[0]) + arm.L2*np.cos(desiredAngles[0]+desiredAngles[1])
            yDesAng = arm.L1*np.sin(desiredAngles[0]) + arm.L2*np.sin(desiredAngles[0]+desiredAngles[1])
            
            
            
            gangliaDesAng = np.ones(2) * 0.5
            gangliaDesAng[0] = utils.changeRange(bg.currActI[0], 0.,1.,shoulderRange[0],shoulderRange[1])
            gangliaDesAng[1] = utils.changeRange(bg.currActI[1], 0.,1.,elbowRange[0],elbowRange[1])
            
            xGangliaDes = arm.L1*np.cos(gangliaDesAng[0]) + arm.L2*np.cos(gangliaDesAng[0]+gangliaDesAng[1])
            yGangliaDes = arm.L1*np.sin(gangliaDesAng[0]) + arm.L2*np.sin(gangliaDesAng[0]+gangliaDesAng[1])
            
            if MOTOR_CORTEX == True:
                mtxDesAng = np.ones(2) * 0.5
                mtxDesAng[0] = utils.changeRange(mt.currI[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                mtxDesAng[1] = utils.changeRange(mt.currI[1], 0.,1.,elbowRange[0],elbowRange[1])    
                
                xmtxDes = arm.L1*np.cos(mtxDesAng[0]) + arm.L2*np.cos(mtxDesAng[0]+mtxDesAng[1])
                ymtxDes = arm.L1*np.sin(mtxDesAng[0]) + arm.L2*np.sin(mtxDesAng[0]+mtxDesAng[1])
            
            if CEREBELLUM == True:
                cerebDesAng = np.ones(2) * 0.5
                cerebDesAng[0] = utils.changeRange(cb.currI[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                cerebDesAng[1] = utils.changeRange(cb.currI[1], 0.,1.,elbowRange[0],elbowRange[1])
                
                xCerebDes = arm.L1*np.cos(cerebDesAng[0]) + arm.L2*np.cos(cerebDesAng[0]+cerebDesAng[1])
                yCerebDes = arm.L1*np.sin(cerebDesAng[0]) + arm.L2*np.sin(cerebDesAng[0]+cerebDesAng[1])
            
            
            
            if epoch == startPlotting:
                
                
                fig1   = plt.figure("Workspace", figsize=(9,8))
                
                text1 = plt.figtext(.02, .72, "epoch = %s" % (0), style='italic', bbox={'facecolor':'yellow'})
                text2 = plt.figtext(.02, .62, "trial = %s" % (0), style='italic', bbox={'facecolor':'lightblue'})
                text3 = plt.figtext(.02, .52, "step = %s" % (0), style='italic', bbox={'facecolor':'lightcoral'})
                text4 = plt.figtext(.02, .42, "reward = %s" % (0), style='italic', bbox={'facecolor':'lightgreen'})
                
                
                ax1    = fig1.add_subplot(111, aspect='equal')
                
                line1, = ax1.plot([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf], 'k-', color = 'pink', linewidth = 10)
                point, = ax1.plot([arm.xEndEf], [arm.yEndEf], 'o', color = 'black' , markersize= 20)
                
                desEnd, = ax1.plot([xDesAng], [yDesAng], 'o', color = 'green' , markersize= 10)
                gangliaOut, = ax1.plot([xGangliaDes], [yGangliaDes], 'o', color = 'blue' , markersize= 10)
            #    noiseEnd, = ax1.plot([noiseX], [noiseY], 'o', color = 'green' , markersize= 10)
                
                if MOTOR_CORTEX ==True:
                    cortexOut, = ax1.plot([xmtxDes], [ymtxDes], 'o', color = 'darkblue' , markersize= 10)
            
                if CEREBELLUM == True:
                    cerebOut, = ax1.plot([xCerebDes], [yCerebDes], 'o', color = 'orange' , markersize= 10)
                
                
                visionLimit = patches.Rectangle(np.array([xVisionRange[0], yVisionRange[0]]), np.sqrt((xVisionRange[0] - xVisionRange[1])**2), np.sqrt((yVisionRange[0] - yVisionRange[1])**2) ,linewidth=1,edgecolor='black',facecolor='lightgrey')
                ax1.add_patch(visionLimit)
                
                ax1.set_xlim([-1.0,1.0])
                ax1.set_ylim([-1.0,1.0])
                
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
                
        
        
        
        
        
        
        
            if epoch >= startPlotting:
                text1.set_text("epoch = %s" % (epoch ))
                point.set_data([arm.xEndEf], [arm.yEndEf]) 
                line1.set_data([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf])
                
                
            for trial in xrange(game.maxTrial):
                
                

                
                
                bg.trialReset(maxStep, cerebellumTime)
                
                if CEREBELLUM== True:
                    cb.trialReset(maxStep)
                    
                
                if trial % 2 == 0:
                    bg.prvRew = 0
                else:
                    bg.prvprvRew = 0
                    
                    
                
                
                
                
                game.setGoal(trial)
                game.goalIndex1(trial)
              #  print trial, game.goalPos
              #  print game.goalIdx
                
              
              
              
              
              
              
              
              
              
                if MULTINET == True:           
                    bg.critW *= 0
                    bg.actW *= 0
                    if CEREBELLUM == True:
                        cb.w *= 0    
                    for i in xrange(len(game.goalList)):
                        
                        if (game.goalPos == game.goalList[i]).all():
                    #       print i
                            bg.critW = bg.multiCritW[:,i].copy()
                        #    print (bg.multiCritW[:,5] == bg.multiCritW[:,4]).all()
                            bg.actW = bg.multiActW[:,:,i].copy()
                            if CEREBELLUM == True:
                                cb.w = cb.multiCerebW[:,:,i].copy()
#                                cb.fwdVisionW = cb.multiFwdVisionW[:,:,i].copy()
                            break
                        
                
                
                if epoch < perfBuff:    
                     T = 1.0 
                else:
                     T = (1.0 - (np.sum(bg.performance[:, game.goalIdx]) / perfBuff))
              #   T = 1 * np.exp(- trial   * 0.1 / float(n_trial))    
               # T = 1. * utils.clipped_exp((-np.sum(bg.performance[:, game.goalIdx]) / perfBuff) * 10.0)    
               # print T, (np.sum(bg.performance[:, game.goalIdx]) / perfBuff) * 10.0   
          #      T = 1.0 - float(epoch)/maxEpoch    
                    
                if VISION == True:
                    if GOAL_VISION == True:
                        netGoalVision[0] = utils.changeRange(game.goalPos[0], xVisionRange[0], xVisionRange[1], 0., 1.)
                        netGoalVision[1] = utils.changeRange(game.goalPos[1], yVisionRange[0], yVisionRange[1], 0., 1.)
                        bg.acquireGoalVision(netGoalVision)
                    #    b = bg.goalVisionRawState.copy()
                        if AGENT_VISION == False:
                            bg.visionState = bg.goalVisionState.copy()
                
                
                            
                
                if epoch >= startPlotting:  
                
                    text2.set_text("trial = %s" % (trial ))
               # text4.set_text("reward = %s" % (bg.rewardCounter[game.goalIdx]))
                    rewardCircle.remove()
                    rewardCircle = plt.Circle((game.goalPos), goalRange, color = 'red') 
                    ax1.add_artist(rewardCircle)
                    plt.pause(0.1)
                    
                    
                    
                    
                for step in xrange(maxStep):
                    
                    
                    
                    if step == 0:
                        if trial == 1 or trial == 3 or trial == 5:
                            if bg.prvRew == 0:
                                break
         #               elif trial == 2 or trial == 4 or trial == 6:
         #                   if bg.prvprvRew == 0:
         #                       break
                            
                            
                    game.prvPos = game.currPos.copy()
                    game.prvVel = game.currVel.copy()
                    game.prvAcc = game.currAcc.copy()
                    
                    
                    
                    if time > 0:
                        bg.prvState = bg.currState.copy()
                        bg.prvCritOut = bg.currCritOut.copy()
                        
                        bg.prvActU = bg.currActU.copy()
                        bg.prvActI = bg.currActI.copy()
                        
                        
                    #    bg.prvLeakedOut = bg.leakedOut.copy()
                        
                        
                        bg.prvLeakedNoise = bg.leakedNoise.copy()
                        bg.prvNoise = bg.currNoise.copy()
                            
                            
                            
                        
                            
                                    
                                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    Torque = arm.PD_controller(np.array([desiredAngles[0],desiredAngles[1]]), Kp  ,Kd) # 18 , 2.1

                    arm.SolveDirectDynamics(Torque[0], Torque[1])
                    
                    game.currPos = np.array([arm.xEndEf,arm.yEndEf])  
                    
                    if time > 0:
                        game.currVel = (utils.distance(game.currPos,game.prvPos)) / arm.dt
                        game.currAcc = (game.currVel - game.prvVel) / arm.dt
                        game.currJerk = (game.currAcc - game.prvAcc) / arm.dt
                                        
                    







                    gangliaDesAng[0] = utils.changeRange(bg.currActI[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                    gangliaDesAng[1] = utils.changeRange(bg.currActI[1], 0.,1., elbowRange[0],elbowRange[1])

                    
                    if CEREBELLUM == True:
                        
                        if CEREBELLUM_DAMAGE == False:

                            cerebDesAng[0] = utils.changeRange(cb.currI[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                            cerebDesAng[1] = utils.changeRange(cb.currI[1], 0.,1., elbowRange[0],elbowRange[1])
                            
                        else:

                            cerebDesAng[0] = utils.changeRange(cb.damageI[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                            cerebDesAng[1] = utils.changeRange(cb.damageI[1], 0.,1., elbowRange[0],elbowRange[1])
                            
                    if MOTOR_CORTEX == True:  
                        
                        mtxDesAng[0] = utils.changeRange(mt.currI[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                        mtxDesAng[1] = utils.changeRange(mt.currI[1], 0.,1., elbowRange[0],elbowRange[1])

                         
        
    




                    
                    
                    
                    if saveData == True:
                        
                        game.goalPositionHistory[:,trial,epoch] = game.goalPos.copy() 
                        
                        if trial%2 == 0:
                            game.goalAnglesHistory[:,step,trial,epoch] = np.array([1.04343374,  2.10975012])
                        elif trial == 1:
                            game.goalAnglesHistory[:,step,trial,epoch] = np.array([1.70435218,  0.77880067])
                        elif trial == 5:
                            game.goalAnglesHistory[:,step,trial,epoch] = np.array([1.1748777,   1.28808818])
                        elif trial == 3:
                            game.goalAnglesHistory[:,step,trial,epoch] = np.array([0.89977078,  1.33238048])
                            
                            
                        game.trialTrajectories[:,step,trial,epoch] = game.currPos.copy()
                        game.trialArmAngles[:,step,trial,epoch] = np.array([arm.theta1,arm.theta2])
                        game.trialGangliaAngles[:,step,trial,epoch]  = gangliaDesAng.copy()
                        
                        if MOTOR_CORTEX == True:
                            game.trialmtxAngles[:,step,trial,epoch] = mtxDesAng.copy()
                        
                        if CEREBELLUM == True:
                            game.trialCerebAngles[:,step,trial,epoch] = cerebDesAng.copy() 
                           # print (game.trialCerebAngles[:,step,trial,epoch] == cerebDesAng).all()
                        
                    #    game.trialVelocity[steptrial,epoch] = game.currVel.copy()
                    #    game.trialAccelleration[step,trial,epoch] = game.currAcc.copy()
                        game.trialJerk[step,trial,epoch] = game.currJerk.copy()  
                        
                        
                        
                        
                        
                        
                        
                        
                    if epoch >= startPlotting:  
                        
                        text3.set_text("step = %s" % (step))
                        text4.set_text("reward =%s" % (bg.rewardCounter[game.goalIdx]))
                        line1.set_data([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf])
                        point.set_data([arm.xEndEf], [arm.yEndEf]) 
                   #     ax1.scatter([arm.xEndEf], [arm.yEndEf])
                                                                   
                        xDesAng = arm.L1*np.cos(desiredAngles[0]) + arm.L2*np.cos(desiredAngles[0]+desiredAngles[1])
                        yDesAng = arm.L1*np.sin(desiredAngles[0]) + arm.L2*np.sin(desiredAngles[0]+desiredAngles[1])
                        desEnd.set_data([xDesAng],[yDesAng])
                                              
                        
                        xGangliaDes = arm.L1*np.cos(gangliaDesAng[0]) + arm.L2*np.cos(gangliaDesAng[0]+gangliaDesAng[1])
                        yGangliaDes = arm.L1*np.sin(gangliaDesAng[0]) + arm.L2*np.sin(gangliaDesAng[0]+gangliaDesAng[1])
                        gangliaOut.set_data( [xGangliaDes] , [yGangliaDes] )
                        
                        
                        if MOTOR_CORTEX == True:
                            xmtxDes = arm.L1*np.cos(mtxDesAng[0]) + arm.L2*np.cos(mtxDesAng[0]+mtxDesAng[1])
                            ymtxDes = arm.L1*np.sin(mtxDesAng[0]) + arm.L2*np.sin(mtxDesAng[0]+mtxDesAng[1])
                            cortexOut.set_data([xmtxDes], [ymtxDes])
                            
            
                        
                        if CEREBELLUM == True:
                            xCerebDes = arm.L1*np.cos(cerebDesAng[0]) + arm.L2*np.cos(cerebDesAng[0]+cerebDesAng[1])
                            yCerebDes = arm.L1*np.sin(cerebDesAng[0]) + arm.L2*np.sin(cerebDesAng[0]+cerebDesAng[1])
                            cerebOut.set_data([xCerebDes], [yCerebDes])
                                
                        plt.pause(0.01) 
                        
                        
                        
                    
                    
                    
                    
                    if PROPRIOCEPTION == True:
                        bg.acquireProprioception(np.array([utils.changeRange(arm.theta1, shoulderRange[0], shoulderRange[1], 0., 1.), utils.changeRange(arm.theta2, elbowRange[0], elbowRange[1], 0., 1.)]))
                        
                        
                        
                    if VISION == True:
                        if AGENT_VISION == True:
                            
                            netAgentVision[0] = utils.changeRange(game.currPos[0], xVisionRange[0], xVisionRange[1], 0., 1.)
                            netAgentVision[1] = utils.changeRange(game.currPos[1], yVisionRange[0], yVisionRange[1], 0., 1.)
                            
                            bg.acquireAgentVision(netAgentVision)
                            if GOAL_VISION == True:
                                bg.visionState = np.hstack([bg.agentVisionState, bg.goalVisionState])
                            else:
                                bg.visionState = bg.agentVisionState.copy()
                                
                                
                    
                    
                    if PROPRIOCEPTION== True:
                        if VISION == True:
                            bg.currState = np.hstack([bg.proprioceptionState, bg.visionState])  
                        else:
                            bg.currState = bg.proprioceptionState.copy()
                    else:
                        if VISION == True:
                            bg.currState = bg.visionState.copy()
                            
                            
                            
                    
                    
                    
                    
                    
                    game.computeDistance()
                    
                    if game.distance < goalRange:
                        
                        
                        
                        if trial == 1 or trial == 3 or trial == 5:
                            
                            if step > 0:
                                if bg.prvRew == 1:
                          #      bg.prvprvRew = 1
                                    bg.actRew = np.e**(-epsi * game.currVel)
            #                        #     bg.spreadCrit()
                                    bg.currCritOut = np.zeros(1)
                        
                        else:
                            if step > 0:
                                bg.prvRew = 1
                                bg.actRew = np.e**(-epsi * game.currVel)

                            #    print bg.actRew
                           #     bg.spreadCrit()
                                bg.currCritOut = np.zeros(1)
                                
                                
                    elif step == maxStep -1:
                        
                        bg.currCritOut = np.zeros(1)
                    
                    else:
                        
                        if step > 0:
                            bg.spreadCrit()   
                    
                    
                    
                    if step > 0:
                    #    print bg.currCritOut , bg.prvCritOut 
                        bg.compSurprise()
               #         print bg.surp

                    
               
               
               
                    if epoch > startPlotting:
                    #    print "actor out", bg.currActOut
                        print "critic out", type(bg.currCritOut), bg.currCritOut , "surp", bg.surp 
                                                
                                                
                    if bg.actRew > 0:
                        arm.stopMove() 
                    
                    
                    
                    
                    
                    
                            
                                                
                    bg.compU()                            
                    bg.spreadAct()       
                    
                    if GANGLIA_NOISE== True:
                        if epoch < startAtaxia:
                            bg.noise(T)                            
                            bg.leakedNoise = utils.limitRange(bg.noiseC4 * bg.leakedNoise + bg.noiseC3 * bg.currNoise, -1.0, 1.0) * T
                          #  bg.leakedNoise = utils.limitRange(bg.leakedNoise + bg.noiseC1 * (bg.noiseC2 * bg.currNoise - bg.leakedNoise), -1.0, 1.0)
                    
                    
                    
                    
                    
                    if MOTOR_CORTEX == True:
                        mt.compU(bg.goalVisionState)
                        mt.compI()
                    
                    
                    
                    
                    
                    if CEREBELLUM == True:
                        
                        cb.compU(bg.currState)
                        
                        

                        if CEREBELLUM_DAMAGE == False:
                #            cb.tau = 1.0
                #            cb.C1 = cb.dT / cb.tau
                #            cb.C2 = 1. - cb.C1
                            
                            cb.spreading()
                 
                        else:
                            
                            if TdCS == False:                   
                                cb.spreading()
                            else:                                 
                                cb.spreadingtDCS(TdCSMag) 
                    
                            cb.tau1 =  np.random.normal(0.0, (damageMag -1))                           
                            if cb.tau1 < 1.0:
                                cb.tau1 = 1.0
                            
                            cb.damageI = (1.0 - cb.dT / cb.tau1) * cb.damageI + (cb.dT / cb.tau1) * cb.currI

                            
                    
                            
                    if MOTOR_CORTEX == True and CEREBELLUM == True:
                        
                        K = 1./3                          
                        if CEREBELLUM_DAMAGE == False:                            
                            netOut = K * bg.currActI + K* mt.currI + K* cb.currI
                        else:
                            netOut = K * bg.currActI + K* mt.currI + K* cb.damageI
                    
                    elif CEREBELLUM == True:
                        K = 1./2
                        if CEREBELLUM_DAMAGE == False:    
                            netOut = K * bg.currActI + K* cb.currI
                        else:
                            netOut = K * bg.currActI + K* cb.damageI
                    else:
                        K = 1
                        netOut = K * bg.currActI


                    

                    desiredAngles[0] = utils.changeRange(utils.limitRange(netOut[0] + bg.leakedNoise[0], 0., 1.) , 0., 1., shoulderRange[0] , shoulderRange[1])
                    desiredAngles[1] = utils.changeRange(utils.limitRange(netOut[1] + bg.leakedNoise[1], 0., 1.), 0., 1.,  elbowRange[0] , elbowRange[1])


                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    if CEREBELLUM == True:
                            
                        bg.netOutBuff[:,step] = netOut.copy()
                        bg.stateBuff[:,step] = bg.currState.copy()
                        
                        if INTRALAMINAR_NUCLEI == True:
                            cb.desOutBuff[:,step] = cb.currI.copy()
                            bg.desOutBuff[:,step] = bg.currActI.copy()
                    




                    if TRAINING == True:
                        
                        if CEREBELLUM_DAMAGE == False:
                        
                            if CEREBELLUM == True: 
                                if (bg.actRew > 0):  
                                    for i in xrange(step + 1): 
                                        cb.trainCb(bg.stateBuff[:,i], bg.netOutBuff[:,i], bg.actRew)
                                        if INTRALAMINAR_NUCLEI == True:
                                            bg.trainAct2(bg.stateBuff[:,i], cb.desOutBuff[:,i], bg.desOutBuff[:,i])
                             
                            
                            if MOTOR_CORTEX == True:
                                if (bg.actRew > 0):
                                    mt.training(bg.goalVisionState, netOut, bg.actRew)
                            
                            
                            if step > 0:        
                                bg.trainCrit()
                                bg.trainAct()
                                
                        
                                    
                            
                              
                    
                    
                    
                    
                    
                    
                    
                    if MULTINET == True:
                        for i in xrange(len(game.goalList)):
                            if (game.goalPos == game.goalList[i]).all():
                                bg.multiCritW[:,i] = bg.critW.copy()
                                bg.multiActW[:,:,i] = bg.actW.copy()
                                if CEREBELLUM == True:
                                    cb.multiCerebW[:,:,i] = cb.w.copy()
                             #       cb.multiFwdVisionW[:,:,i] = cb.fwdVisionW.copy()
                                break
                            
                            
                            
                    time += 1
                
                
                    

                
                    if bg.actRew > 0:
                        epochAccurancy[trial%game.maxTrial] = 1.
                        epochGoalTime[trial%game.maxTrial] = step 
                        bg.performance[epoch%perfBuff,game.goalIdx] = 1.
                        
                        print trial, game.goalPos, step, bg.actRew, np.array([arm.theta1,arm.theta2]), np.round(np.sum(bg.performance[:, game.goalIdx]) / perfBuff , 2)
                        
                        break
                    
               #     if bg.actRew < 0:
                #        epochAccurancy[trial%game.maxTrial] = 0.
                 #       epochGoalTime[trial%game.maxTrial] = maxStep
                   #     bg.performance[epoch%perfBuff,game.goalIdx] = 0.
                    #    print trial, game.goalPos, step, bg.actRew, np.array([arm.theta1,arm.theta2])
                    #    bg.rewardCounter[game.goalIdx] +=1
                  #      break
                    
                    
                if bg.actRew == 0:    
                    epochAccurancy[trial%game.maxTrial] = 0.          
                    epochGoalTime[trial%game.maxTrial] = maxStep 
                    bg.performance[epoch%perfBuff,game.goalIdx] = 0.
                
                
                
                
                
                
                
            epochAvgAcc  = (float(np.sum(epochAccurancy)) / game.maxTrial) * 100
            epochAvgTime = (float(np.sum(epochGoalTime)) / game.maxTrial)
        
            print "epoch" , epoch, "avarage steps", round(epochAvgTime,2)  , "accurancy" , round(epochAvgAcc,2), "%", "seed", seed
        
            avg5EpochAcc[epoch%avgStats] = epochAvgAcc
            avg5EpochTime[epoch%avgStats] = epochAvgTime
                         
            if epoch % avgStats == (avgStats -1):
                finalAvgAcc[(epoch/avgStats)%(maxEpoch/avgStats)] = np.sum(avg5EpochAcc) / avgStats
                finalAvgTime[(epoch/avgStats)%(maxEpoch/avgStats)] = np.sum(avg5EpochTime) / avgStats
                print "******avg 5 epoch" , epoch, "avarage steps", round(np.sum(avg5EpochTime) / avgStats,2), "accurancy" , round(np.sum(avg5EpochAcc) / avgStats,2), "%"
                    
                    
                    
        
        plt.figure(figsize=(120, 4), num=3, dpi=160)
        plt.title('average time in 10 epoch')
        plt.xlim([0, maxEpoch/avgStats])
        plt.ylim([0, maxStep])
        plt.xlabel("epochs")
        plt.ylabel("s")
        plt.xticks(np.arange(0,maxEpoch/avgStats, 10))
        if CEREBELLUM == True:
            if INTRALAMINAR_NUCLEI == True:
                plt.plot(finalAvgTime, label ="Full_Sistem_actETA2=%s_seed=%s" % (bg.ACT_ETA2, seed))
            else:
                plt.plot(finalAvgTime, label ="No_Intralaminar_CbETA=%s_seed=%s" % (cb.cbETA, seed))
        else:
            plt.plot(finalAvgTime, label ="Only_Ganglia_" + "actETA1=" + str(bg.ACT_ETA1) + "critETA=" + str(bg.CRIT_ETA) + "seed=" + str(seed))
        
        plt.legend(loc='lower left')
        
        
        
        plt.figure(figsize=(120, 4), num=4 ,dpi=160)
        plt.title('% accurancy')
        plt.xlim([0,maxEpoch/avgStats])
        plt.ylim([0,101])
        plt.xlabel("epochs")
        plt.ylabel("accurancy %")
        plt.xticks(np.arange(0,maxEpoch/avgStats, 10))
        if CEREBELLUM == True:
            if INTRALAMINAR_NUCLEI == True:
                plt.plot(finalAvgAcc, label ="Full_Sistem_actETA2=%s_seed=%s" % (bg.ACT_ETA2, seed))
            else:
                plt.plot(finalAvgAcc, label ="No_Intralaminar_CbETA=%s_seed=%s" % (cb.cbETA, seed))
        else:
            plt.plot(finalAvgAcc, label ="Only_Ganglia_" + "actETA1=" + str(bg.ACT_ETA1) + "critETA=" + str(bg.CRIT_ETA) + "seed=" + str(seed))
        
        plt.legend(loc='lower left') 
        
        
        
        
        
        if saveData == True:
            mydir = os.getcwd
            os.chdir('/home/adriano/Desktop/target6_2/data/')   
            
                        
            if MOTOR_CORTEX == True and CEREBELLUM == True:
                
                if not os.path.exists(os.curdir + "/fullsystem/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s_mtETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,cb.cbETA,mt.ETA)):
                    os.makedirs(os.curdir + "/intralaminarNuclei/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s_mtETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,cb.cbETA,mt.ETA))
                os.chdir(os.curdir + "/intralaminarNuclei/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s_mtETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,cb.cbETA,mt.ETA))
                
                if CEREBELLUM_DAMAGE == True:
                    if not os.path.exists(os.curdir + "/cerebellumDamage=" + str(damageMag)):
                        os.makedirs(os.curdir + "/cerebellumDamage=" + str(damageMag))
                    os.chdir(os.curdir + "/cerebellumDamage=" + str(damageMag))
                
                np.save("motorcortexweights" % (seed), (mt.w))
                np.save("cerebellumWeights_seed=%s" % (seed), (cb.w))
                np.save("gameTrialCerebAngles_seed=%s" % (seed), game.trialCerebAngles)
                np.save("gameTrialmtxAngles_seed=%s" % (seed), game.trialmtxAngles)
            
            
            
            
            
            
            elif MOTOR_CORTEX == True:
                
                if not os.path.exists(os.curdir + "/nocerebebellum/actETA1=%s_actETA2=%s_critETA=%s_mtETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,mt.ETA)):
                    os.makedirs(os.curdir + "/nocerebebellum/actETA1=%s_actETA2=%s_critETA=%s_mtETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,mt.ETA))
                os.chdir(os.curdir + "/nocerebebellum/actETA1=%s_actETA2=%s_critETA=%s_mtETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,mt.ETA))
                
                np.save("motorcortexweights" % (seed), (mt.w))
                np.save("gameTrialmtxAngles_seed=%s" % (seed), game.trialmtxAngles)
            
            
            
            
            
            
            elif CEREBELLUM == True:
                
                if not os.path.exists(os.curdir + "/nomotorcortex/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,cb.cbETA)):
                    os.makedirs(os.curdir + "/nomotorcortex/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,cb.cbETA,))
                os.chdir(os.curdir + "/nomotorcortex/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,cb.cbETA))    
                
                if CEREBELLUM_DAMAGE == True:
                    if not os.path.exists(os.curdir + "/cerebellumDamage=" + str(damageMag)):
                        os.makedirs(os.curdir + "/cerebellumDamage=" + str(damageMag))
                    os.chdir(os.curdir + "/cerebellumDamage=" + str(damageMag))
                
                np.save("cerebellumWeights_seed=%s" % (seed), (cb.w))
                np.save("gameTrialCerebAngles_seed=%s" % (seed), game.trialCerebAngles)
                
                
                
                
                
            else:
                if not os.path.exists(os.curdir + "/onlyGanglia/actETA1=%s_critETA=%s/" % (bg.ACT_ETA1,bg.CRIT_ETA)):
                    os.makedirs(os.curdir + "/onlyGanglia/actETA1=%s_critETA=%s/" % (bg.ACT_ETA1,bg.CRIT_ETA))
                os.chdir(os.curdir + "/onlyGanglia/actETA1=%s_critETA=%s/" % (bg.ACT_ETA1,bg.CRIT_ETA))
            
            
            
            np.save("actorWeights_seed=%s" % (seed), (bg.actW))
            np.save("criticWeights_seed=%s" % (seed), (bg.critW))
        
            np.save("goalPositionHistory_seed=%s" % (seed), game.goalPositionHistory)
            np.save("goalAnglesHistory_seed=%s" % (seed), game.goalAnglesHistory)
            
            
            
            np.save("gameTrajectories_seed=%s" % (seed), game.trialTrajectories)
            np.save("gameTrialArmAngles_seed=%s" % (seed), game.trialArmAngles)
            np.save("gameTrialGangliaAngles_seed=%s" % (seed), game.trialGangliaAngles)
                        
            
        #    np.save("gameVelocity_seed=%s" % (seed), game.trialVelocity)
        #    np.save("gameAccelleration_seed=%s" % (seed), game.trialAccelleration)
            np.save("gameJerk_seed=%s" % (seed), game.trialJerk)
            
            
    
                    
                            
                    
                                    
                                    
                                    
                    
                                                
                                                
                            
                                
                        
                        
                    
















