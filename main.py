# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 21:01:09 2018

@author: Alex
"""

from arm import Arm
from basalGanglia import actorCritic
from cerebellum import Cerebellum 
from games import armReaching6targets


import numpy as np
import copy as copy
import matplotlib.pyplot as plt
import utilities as utils
import os




shoulderRange = np.array([-1.0, np.pi])
elbowRange = np.array([0.0, np.pi ])







TRAINING = True
GANGLIA_NOISE = True
gangliaTime = 12
perfBuff = 20

MULTINET = False




CEREBELLUM = False
CEREBELLUM_DAMAGE = False
damageMag = 5.
INTRALAMINAR_NUCLEI = False
cerebellumTime = 1

if CEREBELLUM == True:
    K = 0.5
else:
    K = 1.






Kp = 5.0
Kd = 0.5





VISION = True
GOAL_VISION = True
AGENT_VISION = False

PROPRIOCEPTION = True
SHOULDER_PROPRIOCEPTION = True
ELBOW_PROPRIOCEPTION = True









               





avgStats = 10

loadData = False
saveData = True










goalRange = 0.01
maxSeed = 1
maxEpoch = 100
maxStep = 300

startPlotting = 0#89#00#180#650#620

startAtaxia = 90




if __name__ == "__main__":
    




    for seed in xrange(maxSeed):
        
        CEREBELLUM_DAMAGE = False
        
        
        
        
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
            cb.init(MULTINET, game.goalList, bg.currState)
            
            




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
            
         #   print CEREBELLUM_DAMAGE
            
            if CEREBELLUM == True:
                if epoch >= startAtaxia:
                    CEREBELLUM_DAMAGE = True
            
            time = 0
            
            
            
            bg.epochReset(maxStep, cerebellumTime)
            arm.epochReset(shoulderRange, elbowRange)
            game.epochReset()
            game.currPos = np.array([arm.xEndEf,arm.yEndEf])  
            
            
            
            if CEREBELLUM == True:
                cb.epochReset()
            
            
            netOut = np.ones(2) * 0.5
            noisedOut = np.ones(2) * 0.5
            prvNoisedOut = np.ones(2) * 0.5
            
            
                            

            xGangliaDes = utils.changeRange(bg.leakedOut[0], 0.,1.,shoulderRange[0],shoulderRange[1])
            yGangliaDes = utils.changeRange(bg.leakedOut[1], 0.,1., elbowRange[0],  elbowRange[1])
            
            noiseDesAng = np.zeros(2)
            
            noiseDesAng[0] = utils.changeRange(bg.currNoise[0], 0.,1.,shoulderRange[0],shoulderRange[1])
            noiseDesAng[1] = utils.changeRange(bg.currNoise[1], 0.,1.,shoulderRange[0],shoulderRange[1])
            
            noiseX = arm.L1*np.cos(noiseDesAng[0]) + arm.L2*np.cos(noiseDesAng[0]+noiseDesAng[1]) 
            noiseY = arm.L1*np.sin(noiseDesAng[0]) + arm.L2*np.sin(noiseDesAng[0]+noiseDesAng[1])
            
            
            if CEREBELLUM == True:
                cerebDesAng = np.ones(2) * 0.5
                cerebDesAng[0] = utils.changeRange(cb.currOut[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                cerebDesAng[1] = utils.changeRange(cb.currOut[1], 0.,1.,elbowRange[0],elbowRange[1])
            
            
            
            desiredAngles = np.ones(2) * 0.5
            desiredAngles[0] = utils.changeRange(desiredAngles[0], 0.,1.,shoulderRange[0],shoulderRange[1])
            desiredAngles[1] = utils.changeRange(desiredAngles[1], 0.,1., elbowRange[0],elbowRange[1])
            prvDesAngles = desiredAngles.copy()
            
            xDesAng = arm.L1*np.cos(desiredAngles[0]) + arm.L2*np.cos(desiredAngles[0]+desiredAngles[1])
            yDesAng = arm.L1*np.sin(desiredAngles[0]) + arm.L2*np.sin(desiredAngles[0]+desiredAngles[1])
            
            
            
            gangliaDesAng = np.ones(2) * 0.5
            gangliaDesAng[0] = utils.changeRange(bg.leakedOut[0], 0.,1.,shoulderRange[0],shoulderRange[1])
            gangliaDesAng[1] = utils.changeRange(bg.leakedOut[1], 0.,1.,elbowRange[0],elbowRange[1])
            
            xGangliaDes = arm.L1*np.cos(gangliaDesAng[0]) + arm.L2*np.cos(gangliaDesAng[0]+gangliaDesAng[1])
            yGangliaDes = arm.L1*np.sin(gangliaDesAng[0]) + arm.L2*np.sin(gangliaDesAng[0]+gangliaDesAng[1])
            
            
            
            if CEREBELLUM == True:
                cerebDesAng = np.ones(2) * 0.5
                cerebDesAng[0] = utils.changeRange(cb.currOut[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                cerebDesAng[1] = utils.changeRange(cb.currOut[1], 0.,1.,elbowRange[0],elbowRange[1])
                
                xCerebDes = arm.L1*np.cos(cerebDesAng[0]) + arm.L2*np.cos(cerebDesAng[0]+cerebDesAng[1])
                yCerebDes = arm.L1*np.sin(cerebDesAng[0]) + arm.L2*np.sin(cerebDesAng[0]+cerebDesAng[1])
            
            
            
            if epoch == startPlotting:
                
                
                fig1   = plt.figure("Workspace", figsize=(9,8))
                
                text1 = plt.figtext(.02, .72, "epoch = %s" % (0), style='italic', bbox={'facecolor':'yellow'})
                text2 = plt.figtext(.02, .62, "trial = %s" % (0), style='italic', bbox={'facecolor':'lightblue'})
                text3 = plt.figtext(.02, .52, "step = %s" % (0), style='italic', bbox={'facecolor':'lightcoral'})
                text4 = plt.figtext(.02, .42, "reward = %s" % (0), style='italic', bbox={'facecolor':'lightgreen'})
                
                
                ax1    = fig1.add_subplot(111)
                
                line1, = ax1.plot([0, arm.xElbow, arm.xEndEf], [0, arm.yElbow, arm.yEndEf], 'k-', color = 'pink', linewidth = 10)
                point, = ax1.plot([arm.xEndEf], [arm.yEndEf], 'o', color = 'black' , markersize= 20)
                desEnd, = ax1.plot([xDesAng], [yDesAng], 'o', color = 'green' , markersize= 10)
                gangliaOut, = ax1.plot([xGangliaDes], [yGangliaDes], 'o', color = 'blue' , markersize= 10)
            #    noiseEnd, = ax1.plot([noiseX], [noiseY], 'o', color = 'green' , markersize= 10)
                
                if CEREBELLUM == True:
                    cerebOut, = ax1.plot([xCerebDes], [yCerebDes], 'o', color = 'orange' , markersize= 10)
                
                
                ax1.set_xlim([-0.75,0.75])
                ax1.set_ylim([-0.5,0.75])
                
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
                
              #  print arm.theta1, arm.theta2
                
           #     
           #     if epoch >= startAtaxia +1:
           #         arm.setEffPosition(game.prvGoalPos)
                
              #  time = 0

             #   bg.epochReset(maxStep, cerebellumTime)
            #    if CEREBELLUM == True:
           #         cb.epochReset()
          #          
          #      arm.epochReset()
                
          #      else:
                
                bg.trialReset(maxStep, cerebellumTime)
                
                if trial % 2 == 0:
                    bg.prvRew = 0

                
                
              
              
              
             #   if trial ==2:
             #       bg.rew0 = 0
             #   elif trial == 0:
             #       bg.rew1 = 0
                
                
                
                
                
                
            #    if epoch < 10000:
              #  reachType = epoch%3
             #   else:
              #      reachType=(epoch%29)/10
            #    print "reach" ,reachType

            #    reachType = 0
               # print reachType
               
               # if CEREBELLUM_DAMAGE == True:
               #     if trial !=0 :
               #         game.prvGoalPos = game.goalPos.copy()
                #        arm.setEffPosition(game.prvGoalPos)
                #        bg.epochReset(maxStep, cerebellumTime)
                 #       cb.epochReset()
                    

                
                game.setGoal(trial)
                game.goalIndex0(trial)
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
                    T = (1.0 - (np.sum(bg.performance[:, game.goalIdx]) / perfBuff)) #+0.001
                 





           #     print T
#                if epoch <500:
#                    T = 1.0 
#                else:
           #     T = 1.0 - float(epoch)/maxEpoch
            #    print T
                
                
                
                
                
                if VISION == True:
                    if GOAL_VISION == True:
                        bg.acquireGoalVision(utils.changeRange(game.goalPos, -1, 1., 0., 1.))
                        if AGENT_VISION == False:
                            bg.visionState = bg.goalVisionState.copy()
                        
                
                
                

            
            
            
            
            
            
                if epoch >= startPlotting:  
                
                    text2.set_text("trial = %s" % (trial ))
               # text4.set_text("reward = %s" % (bg.rewardCounter[game.goalIdx]))
                    rewardCircle.remove()
                    rewardCircle = plt.Circle((game.goalPos), goalRange, color = 'red') 
                    ax1.add_artist(rewardCircle)
                    plt.pause(0.1)
                    
                    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                "INIT STEP ITERATION"
                for step in xrange(maxStep):
                    
                    if step == 0:
                        if trial == 1 or trial == 3 or trial == 5:
                            if bg.prvRew == 0:
                                break
                    
                    
                    game.prvPos = game.currPos.copy()
                    game.prvVel = game.currVel.copy()
                    game.prvAcc = game.currAcc.copy()
                    
                    
                    
                    
                    

                    if time > (gangliaTime - 1):    
                        if time % gangliaTime == 0:  
                            bg.prvActOut = bg.currActOut.copy()
                            bg.prvLeakedOut = bg.leakedOut.copy()
                            bg.prvCritOut = bg.currCritOut.copy() 
                            bg.prv5State = bg.currState.copy()
                       #     bg.prv5BiasedState = bg.currState.copy()
                            bg.prvLeakedNoise = bg.leakedNoise.copy()
                            bg.prvNoise = bg.currNoise.copy()
                    
                    
                    
                    
                    
                    if CEREBELLUM == True:
                        
                        if time > cerebellumTime -1:
                            
                            if time % cerebellumTime ==0:
                                
                                bg.prvState = bg.currState.copy()                          
                                prvDesAngles = desiredAngles.copy()
                                bg.desOutBuff[:,step%cerebellumTime] = desiredAngles.copy() #prvDesAngles.copy()  
                                bg.stateBuff[:,step%cerebellumTime] = bg.prvState.copy()
                                cb.prvOut = cb.leakedOut.copy()
                            

                                if INTRALAMINAR_NUCLEI == True: 
                                    bg.prvTrainOut = bg.trainOut.copy()
                                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    Torque = arm.PD_controller(np.array([desiredAngles[0],desiredAngles[1]]), Kp  ,Kd) # 18 , 2.1

                    arm.SolveDirectDynamics(Torque[0], Torque[1])
                    
                    game.currPos = np.array([arm.xEndEf,arm.yEndEf])  
                    
                    if step > 0:
                        game.currVel = (utils.distance(game.currPos,game.prvPos)) / arm.dt
                     #   print game.currVel
                        game.currAcc = (game.currVel - game.prvVel) / arm.dt
                        game.currJerk = (game.currAcc - game.prvAcc) / arm.dt
                                        
                     
                        
                                        
      

                    
                                    
                                    
                                   
                                    
                                    
                    if CEREBELLUM == True:
                        if CEREBELLUM_DAMAGE == False:
                            
                            gangliaDesAng[0] = utils.changeRange(bg.leakedOut[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                            gangliaDesAng[1] = utils.changeRange(bg.leakedOut[1], 0.,1., elbowRange[0],elbowRange[1]) 
                            
                            
                            
                            cerebDesAng[0] = utils.changeRange(cb.leakedOut[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                            cerebDesAng[1] = utils.changeRange(cb.leakedOut[1], 0.,1., elbowRange[0],elbowRange[1])
                            
                        else:
                            
                            gangliaDesAng[0] = utils.changeRange(bg.leakedOut[0] , 0.,1.,shoulderRange[0],shoulderRange[1])
                            gangliaDesAng[1] = utils.changeRange(bg.leakedOut[1] , 0.,1., elbowRange[0],elbowRange[1]) 
                            
                            cerebDesAng[0] = utils.changeRange(cb.leakedOut[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                            cerebDesAng[1] = utils.changeRange(cb.leakedOut[1], 0.,1., elbowRange[0],elbowRange[1])
                                            
                    else:
                        gangliaDesAng[0] = utils.changeRange(bg.leakedOut[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                        gangliaDesAng[1] = utils.changeRange(bg.leakedOut[1], 0.,1., elbowRange[0],elbowRange[1]) 
                         
                                    
                    
                    
                    
                    
                    
                    
                    
                    
                    if saveData == True:
                        
                        game.goalPositionHistory[:,trial,epoch] = game.goalPos.copy() 
                        
                        if trial%2 == 0:
                            game.goalAnglesHistory[:,step,trial,epoch] = np.array([0.962333868378, 2.29018598059])
                        elif trial == 1:
                            game.goalAnglesHistory[:,step,trial,epoch] = np.array([1.10853968707, 1.46258215453])
                        elif trial == 5:
                            game.goalAnglesHistory[:,step,trial,epoch] = np.array([0.732029921665, 1.55391175155])
                        elif trial == 3:
                            game.goalAnglesHistory[:,step,trial,epoch] = np.array([1.64159851167, 1.05171498095])
                            
                            
                        game.trialTrajectories[:,step,trial,epoch] = game.currPos.copy()
                        game.trialArmAngles[:,step,trial,epoch] = np.array([arm.theta1,arm.theta2])
                        game.trialGangliaAngles[:,step,trial,epoch]  = gangliaDesAng.copy()
                        
                        if CEREBELLUM == True:
                            game.trialCerebAngles[:,step,trial,epoch] = cerebDesAng.copy() 
                        
                        
                        game.trialVelocity[step,trial,epoch] = game.currVel.copy()
                        game.trialAccelleration[step,trial,epoch] = game.currAcc.copy()
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
                        
                        noiseDesAng[0] = utils.changeRange(bg.currNoise[0], 0.,1.,shoulderRange[0],shoulderRange[1])
                        noiseDesAng[1] = utils.changeRange(bg.currNoise[1], 0.,1.,elbowRange[0],elbowRange[1])
            #
                        noiseX = arm.L1*np.cos(noiseDesAng[0]) + arm.L2*np.cos(noiseDesAng[0]+noiseDesAng[1]) 
                        noiseY = arm.L1*np.sin(noiseDesAng[0]) + arm.L2*np.sin(noiseDesAng[0]+noiseDesAng[1])

                        
                        if CEREBELLUM == True:
                            xCerebDes = arm.L1*np.cos(cerebDesAng[0]) + arm.L2*np.cos(cerebDesAng[0]+cerebDesAng[1])
                            yCerebDes = arm.L1*np.sin(cerebDesAng[0]) + arm.L2*np.sin(cerebDesAng[0]+cerebDesAng[1])
                            cerebOut.set_data([xCerebDes], [yCerebDes])
                                
                        plt.pause(0.01) 
                        

                    
                    
                    
                
                
                
                
                    
                
                    if PROPRIOCEPTION == True:
                        bg.acquireProprioception(np.array([utils.changeRange(arm.theta1, shoulderRange[0], shoulderRange[1], 0., 1.), utils.changeRange(arm.theta2, elbowRange[0], elbowRange[1], 0., 1.)]))
                
                     #   a = bg.proprioceptionState.copy()
                
                
                
                
                
                
              #      if PROPRIOCEPTION == True:
                
           #             if SHOULDER_PROPRIOCEPTION== True:
           #                 bg.acquireShoulderState(utils.changeRange(arm.theta1, shoulderRange[0], shoulderRange[1], 0., 1.))
           #                 if ELBOW_PROPRIOCEPTION== True:
           #                     bg.acquireElbowState(utils.changeRange(arm.theta2, elbowRange[0], elbowRange[1], 0., 1.))
           #                     bg.proprioceptionState = np.hstack([bg.shoulderState, bg.elbowState])
           #                 else:
           #                     bg.proprioceptionState = bg.shoulderState.copy()
                    
     #               a = bg.elbowState.copy()
     #               b = bg.shoulderState.copy()
                    
                    
                    if VISION == True:
                        if AGENT_VISION == True:
                            bg.acquireAgentVision(utils.changeRange(game.currPos, -1, 1., 0., 1.))
                            if GOAL_VISION == True:
                                bg.visionState = np.hstack([bg.agentVisionState, bg.goalVisionState])
                            else:
                                bg.visionState = bg.agentVisionState.copy()
                                
                        a = bg.visionState.copy()
                                
                                
                    
                    
                    if PROPRIOCEPTION== True:
                        if VISION == True:
                            bg.currState = np.hstack([bg.proprioceptionState, bg.visionState])  
                        else:
                            bg.currState = bg.proprioceptionState.copy()
                    else:
                        if VISION == True:
                            bg.currState = bg.visionState.copy()
                            
                            
                #    bg.biasedState = np.hstack([bg.bias, bg.currState])
                        
                    
      

                 #   c = bg.goalVisionRawState.copy()
                 
                 
                 
                 
                 
                 
                    
                    
                    
                    
                 
                 
                    game.computeDistance()
                    
                    if game.distance < goalRange*1.:
                        
                        if trial == 1 or trial == 3 or trial == 5:
                            if step > gangliaTime -1:
                                if bg.prvRew == 1:
                                    bg.actRew = 1                          
                                    bg.currCritOut = np.zeros(1)
                        else:
                            if step > gangliaTime -1:
                                bg.prvRew = 1
                                bg.actRew = 1                          
                                bg.currCritOut = np.zeros(1)
                    
                    else:
                        
                        if step > gangliaTime -1:
                            if time % gangliaTime == 0:
                                bg.spreadCrit()   
                    
                    
                    
                    if step > gangliaTime -1:
                        if (time % gangliaTime == 0 or bg.actRew == 1):
                            bg.compSurprise() 
                   #         if bg.actRew ==1:
                   #             print bg.surp

                    
                    if epoch > startPlotting:
                        print "actor out", bg.currActOut
                        print "critic out", type(bg.currCritOut), bg.currCritOut , "surp", bg.surp 
                        
                    
              
              
              
              
              
              
              
              
              
              
              
                    
                    
                    
                    
                    
                    
                    if CEREBELLUM == True:
                        
                        if time % gangliaTime == 0:
                        #    print "step", step, 1
                            bg.spreadAct()       
                            if GANGLIA_NOISE== True:
                                if epoch < startAtaxia:
                                    bg.noise(T)
                        
                        bg.leakedNoise = utils.limitRange(bg.noiseC4 * bg.leakedNoise + bg.noiseC3 * bg.currNoise, -1.0, 1.0) * T
                        bg.leakedOut = utils.limitRange(bg.actorC2 * bg.leakedOut + bg.actorC1 * bg.currActOut, 0.0, 1.0)

                        if time % cerebellumTime == 0:        
                            cb.spreading(bg.currState) 
                            
                            if CEREBELLUM_DAMAGE == False:
                                cb.tau = 1.
                                cb.C1 = cb.dT / cb.tau
                                cb.C2 = 1. - cb.C1
                                cb.leakedOut = utils.limitRange(cb.C2 * cb.leakedOut + cb.C1 * cb.currOut, 0.0, 1.0)
                            else:
                                cb.tau = copy.deepcopy(damageMag)
                                cb.C1 = cb.dT / cb.tau
                                cb.C2 = 1. - cb.C1
                     
                                cb.leakedOut = utils.limitRange(cb.C2 * cb.leakedOut + cb.C1 * cb.currOut, 0.0, 1.0)
                            
                            
                            if INTRALAMINAR_NUCLEI == True:
                                bg.trainOut = utils.sigmoid(np.dot(bg.actW.T, bg.currState))   
                        
                        
                        
                            
                        netOut = (K * bg.leakedOut) + ((1-K) * cb.leakedOut)
                    
                                               
                    
                    
                    
                    else:
                        if time % gangliaTime == 0:
               #             print "******************" , step
                            bg.spreadAct()
                            
                            if GANGLIA_NOISE== True:
                                if epoch < startAtaxia:
                                    bg.noise(T)
                        
                        bg.leakedNoise = utils.limitRange(bg.noiseC4 * bg.leakedNoise + bg.noiseC3 * bg.currNoise, -1.0, 1.0) * T
                        bg.leakedOut = utils.limitRange(bg.actorC2 * bg.leakedOut + bg.actorC1 * bg.currActOut, 0.0, 1.0)
                        
                  #      bg.leakedNoise =  utils.limitRange(bg.leakedNoise + bg.noiseC3 * (bg.noiseC4 * bg.currNoise - bg.leakedNoise), 0.0, 1.0) * T
                 #       bg.leakedOut =  utils.limitRange(bg.leakedOut + bg.actorC1 * (bg.actorC2 * bg.currActOut - bg.leakedOut), 0.0, 1.0) 
                        
                        
                        
                        netOut = K * bg.leakedOut
                        
                 #   bg.currNoise[0] = 0.
                    
                        
                #    print netOut
                  #  print bg.currNoise    
               #     print bg.currActOut
                    
                    desiredAngles[0] = utils.changeRange(utils.limitRange(netOut[0] + bg.leakedNoise[0], 0., 1.) , 0., 1., shoulderRange[0] , shoulderRange[1])
                    desiredAngles[1] = utils.changeRange(utils.limitRange(netOut[1] + bg.leakedNoise[1], 0., 1.), 0., 1.,  elbowRange[0] , elbowRange[1])
                 
                 
                 
               #     desiredAngles[0] = utils.changeRange(( (1.0 -T) * netOut[0]) + (T * bg.currNoise[0]) , 0., 1., shoulderRange[0] , shoulderRange[1])
               #     desiredAngles[1] = utils.changeRange(( (1.0 -T) * netOut[1]) + (T * bg.currNoise[1]) , 0., 1.,  elbowRange[0] , elbowRange[1])
                    
                    
                    
                    
                              
                            
                            
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    if TRAINING == True:
                        
                        if time > (gangliaTime -1): 
                            if (time % gangliaTime == 0 or bg.actRew == 1):       
                                bg.trainCrit()
                                bg.trainAct()
                                
                              #  if INTRALAMINAR_NUCLEI == True:
                              #      if time % cerebellumTime == 0:
                              #          bg.actW += bg.ACT_ETA1 * bg.surp * np.outer(bg.prv5State , bg.prvNoise) +  bg.ACT_ETA2 * np.outer(bg.prvState, (cb.prvOut - bg.prvTrainOut))
                              #  else:
                              #      bg.trainAct()  
                                
                        if CEREBELLUM == True:
                            if INTRALAMINAR_NUCLEI== True:
                                if time > cerebellumTime -1:
                                    if time % cerebellumTime == 0:
                                        if CEREBELLUM_DAMAGE == False:
                                            bg.trainAct2(cb.prvOut,T)
                                        else:
                                            bg.trainAct2(cb.prvOut * (1.0 - damageMag), T)
                                
                            
                            
                            if (bg.actRew == 1):
                                for i in xrange((step + 1)/cerebellumTime): 
                                 #   print i
                                    bg.desOutBuff[0,i] = utils.changeRange(bg.desOutBuff[0,i], shoulderRange[0], shoulderRange[1], 0., 1.)
                                    bg.desOutBuff[1,i] = utils.changeRange(bg.desOutBuff[1,i],  elbowRange[0], elbowRange[1], 0., 1.)
                                    cb.trainCb(bg.stateBuff[:,i], bg.desOutBuff[:,i])
                                
                    
                    
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
                
                
                    

                
                    if bg.actRew == 1:
                        epochAccurancy[trial%game.maxTrial] = 1.
                        epochGoalTime[trial%game.maxTrial] = step 
                        bg.performance[epoch%perfBuff,game.goalIdx] = 1.
                        print trial, game.goalPos, step , arm.theta1, arm.theta2
                        bg.rewardCounter[game.goalIdx] +=1
                        break
                
                
                
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
                   
            if CEREBELLUM == True:
                
                
                if INTRALAMINAR_NUCLEI == True:    
                    
                    if not os.path.exists("C:\Users/Alex/Desktop/targets6/data/intralaminarNuclei/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,cb.cbETA)):
                        os.makedirs("C:\Users/Alex/Desktop/targets6/data/intralaminarNuclei/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,cb.cbETA))
                    os.chdir("C:\Users/Alex/Desktop/targets6/data/intralaminarNuclei/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s/" % (bg.ACT_ETA1,bg.ACT_ETA2,bg.CRIT_ETA,cb.cbETA))
                    
                
                else:
                    if not os.path.exists("C:\Users/Alex/Desktop/targets6/data/onlyCerebellum/actETA1=%s_critETA=%s_cbETA=%s/" % (bg.ACT_ETA1,bg.CRIT_ETA,cb.cbETA)):
                        os.makedirs("C:\Users/Alex/Desktop/targets6/data/onlyCerebellum/actETA1=%s_critETA=%s_cbETA=%s/" % (bg.ACT_ETA1,bg.CRIT_ETA,cb.cbETA))
                    os.chdir("C:\Users/Alex/Desktop/targets6/data/onlyCerebellum/actETA1=%s_critETA=%s_cbETA=%s/" % (bg.ACT_ETA1,bg.CRIT_ETA,cb.cbETA))
                
                
                if CEREBELLUM_DAMAGE == True:
                    if not os.path.exists(os.curdir + "/cerebellumDamage=" + str(damageMag)):
                        os.makedirs(os.curdir + "/cerebellumDamage=" + str(damageMag))
                    os.chdir(os.curdir + "/cerebellumDamage=" + str(damageMag))
                
                
                np.save("cerebellumWeights_seed=%s" % (seed), (cb.w))
                np.save("gameTrialCerebAngles_seed=%s" % (seed), game.trialCerebAngles)
                
                
            else:
                if not os.path.exists("C:\Users/Alex/Desktop/targets6/data/onlyGanglia/actETA1=%s_critETA=%s/" % (bg.ACT_ETA1,bg.CRIT_ETA)):
                    os.makedirs("C:\Users/Alex/Desktop/targets6/data/onlyGanglia/actETA1=%s_critETA=%s/" % (bg.ACT_ETA1,bg.CRIT_ETA))
                os.chdir("C:\Users/Alex/Desktop/targets6/data/onlyGanglia/actETA1=%s_critETA=%s/" % (bg.ACT_ETA1,bg.CRIT_ETA))
            
            
            
            np.save("actorWeights_seed=%s" % (seed), (bg.actW))
            np.save("criticWeights_seed=%s" % (seed), (bg.critW))
        
        
            np.save("goalPositionHistory_seed=%s" % (seed), game.goalPositionHistory)
            np.save("goalAnglesHistory_seed=%s" % (seed), game.goalAnglesHistory)
            
            
            
            np.save("gameTrajectories_seed=%s" % (seed), game.trialTrajectories)
            np.save("gameTrialArmAngles_seed=%s" % (seed), game.trialArmAngles)
            np.save("gameTrialGangliaAngles_seed=%s" % (seed), game.trialGangliaAngles)
            
            
            np.save("gameVelocity_seed=%s" % (seed), game.trialVelocity)
            np.save("gameAccelleration_seed=%s" % (seed), game.trialAccelleration)
            np.save("gameJerk_seed=%s" % (seed), game.trialJerk)
        
        #os.close(os.curdir)
 
                
                
                
                
                