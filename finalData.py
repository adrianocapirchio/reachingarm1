# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:34:14 2018

@author: Alex
"""

import os
import numpy as np
import copy as copy
import matplotlib.pyplot as plt
import utilities as utils




CEREBELLUM = True
INTRALAMINAR_NUCLEI = False

ATAXIA = True
damageMag = 5.0


goalRange = 0.02

maxStep = 150
maxTrial = 7
maxEpoch = 600
seed =0
maxSeed = 5



actETA1 = 1.0 * 10 ** ( -1)
actETA2 = 1.0 * 10 ** (- 6)
critETA = 1.0 * 10 ** ( -5)
cbETA   = 1.0 * 10 ** ( -1)



n = 20
startAtaxia = 580



if CEREBELLUM == True:
    if INTRALAMINAR_NUCLEI == True:              
        os.chdir("C:\Users/Alex/Desktop/targets6/data/intralaminarNuclei/actETA1=%s_actETA2=%s_critETA=%s_cbETA=%s/" % (actETA1,actETA2,critETA,cbETA))
    else:
        os.chdir("C:\Users/Alex/Desktop/targets6/data/onlyCerebellum/actETA1=%s_critETA=%s_cbETA=%s/" % (actETA1,critETA,cbETA))
    
    if ATAXIA == True:
        os.chdir(os.curdir + "/cerebellumDamage=" + str(damageMag) + "/")
        
            
else:
    os.chdir("C:\Users/Alex/Desktop/targets6/data/onlyGanglia/actETA1=%s_critETA=%s/" % (actETA1,critETA))









allGoalPosition = np.zeros([2, maxTrial, maxEpoch, maxSeed])
allGoalAngles = np.zeros([2, maxStep, maxTrial, maxEpoch, maxSeed])


allArmAngles = np.zeros([2, maxStep, maxTrial, maxEpoch, maxSeed])
allTrajectories = np.zeros([2, maxStep, maxTrial, maxEpoch, maxSeed])
allGangliaAngles = np.zeros([2, maxStep, maxTrial, maxEpoch, maxSeed])

if CEREBELLUM == True:
    allCerebAngles = np.zeros([2, maxStep, maxTrial, maxEpoch, maxSeed])



allVelocity = np.zeros([maxStep, maxTrial, maxEpoch, maxSeed])
allAccelleration = np.zeros([maxStep, maxTrial, maxEpoch, maxSeed])
allJerk = np.zeros([maxStep, maxTrial, maxEpoch, maxSeed])

allLinIdx = np.zeros([maxTrial,maxEpoch,maxSeed])
allAsyIdx = np.zeros([maxTrial,maxEpoch,maxSeed])
allSmoIdx = np.zeros([maxTrial,maxEpoch,maxSeed])



for seed in xrange (maxSeed):
    
    allTrajectories[:,:,:,:,seed]              = np.load("gameTrajectories_seed=%s.npy" %(seed))
    allArmAngles[:,:,:,:,seed]                 = np.load("gameTrialArmAngles_seed=%s.npy" % (seed))
    allGangliaAngles[:,:,:,:,seed]             = np.load("gameTrialGangliaAngles_seed=%s.npy" % (seed))

    if CEREBELLUM == True:
        allCerebAngles[:,:,:,:,seed]           = np.load("gameTrialCerebAngles_seed=%s.npy" %(seed)) 

    allGoalPosition[:,:,:,seed]              = np.load("goalPositionHistory_seed=%s.npy" %(seed))
    allGoalAngles[:,:,:,:,seed]                = np.load("goalAnglesHistory_seed=%s.npy" %(seed))
    allVelocity[:,:,:,seed]                  = np.load("gameVelocity_seed=%s.npy" %(seed) )
    allAccelleration[:,:,:,seed]             = np.load("gameAccelleration_seed=%s.npy" %(seed))
    allJerk[:,:,:,seed]                   = np.load("gameJerk_seed=%s.npy" %(seed))























linearityIndex = np.zeros(1)
asimmetryIndex = np.zeros(1)
smoothnessIndex = np.zeros(1)


reach1LinIdx = np.zeros([n,maxSeed])
reach1AsyIdx = np.zeros([n,maxSeed])
reach1SmoIdx = np.zeros([n,maxSeed])


normalAVGLinIdx = np.zeros(n)
ataxiaAVGLinIdx = np.zeros(n)

normalAVGAsyIdx = np.zeros(n)
ataxiaAVGAsyIdx = np.zeros(n)

normalAVGSmoIdx = np.zeros(n)
ataxiaAVGSmoIdx = np.zeros(n)









for seed in xrange(maxSeed):
    
    for epoch in xrange(maxEpoch):
        
        for trial in xrange(maxTrial):
            
            
            gameGoalPos = allGoalPosition[:,trial,epoch,seed].copy()
            
            
            trialArmAngles = allArmAngles[:,:,trial,epoch,seed].copy()
            trialGoalAngles = allGoalAngles[:,:,trial,epoch,seed].copy()
            trialGangliaAngles = allGangliaAngles[:,:,trial,epoch,seed].copy()
            
            if CEREBELLUM == True:
                trialCerebAngles = allCerebAngles[:,:,trial,epoch,seed].copy()
            
            trialTraj = allTrajectories[:,:,trial,epoch,seed].copy()
            trialVelocity = allVelocity[:,trial,epoch,seed].copy()
            trialAccelleration = allAccelleration[:,trial,epoch,seed].copy()
            trialJerk = allJerk[:,trial,epoch,seed].copy()
            
            linearityIndex = utils.linearityIndex(trialTraj, gameGoalPos, goalRange)
            smoothnessIndex = utils.smoothnessIndex(trialJerk)
            asimmetryIndex = utils.asimmetryIndex(trialVelocity)
            
            allLinIdx[trial,epoch,seed] = copy.deepcopy(linearityIndex)
            allAsyIdx[trial,epoch,seed] = copy.deepcopy(asimmetryIndex)
            allSmoIdx[trial,epoch,seed] = copy.deepcopy(smoothnessIndex)
            




"GET STAT DATA FOR ANALISYS" 
 
"REACH 1"         
statNormalReach1LinIdx = allLinIdx[1,startAtaxia -n : startAtaxia,:]  
statNormalReach1AsyIdx = allAsyIdx[1,startAtaxia -n : startAtaxia,:]
statNormalReach1SmoIdx = allSmoIdx[1,startAtaxia -n : startAtaxia,:] 

statAtaxiaReach1LinIdx = allLinIdx[1,startAtaxia : startAtaxia +n,:]  
statAtaxiaReach1AsyIdx = allAsyIdx[1,startAtaxia : startAtaxia +n,:]
statAtaxiaReach1SmoIdx = allSmoIdx[1,startAtaxia : startAtaxia +n,:]  

"REACH 3"
statNormalReach3LinIdx = allLinIdx[3,startAtaxia -n : startAtaxia,:]  
statNormalReach3AsyIdx = allAsyIdx[3,startAtaxia -n : startAtaxia,:]
statNormalReach3SmoIdx = allSmoIdx[3,startAtaxia -n : startAtaxia,:]  

statAtaxiaReach3LinIdx = allLinIdx[3,startAtaxia : startAtaxia +n,:]  
statAtaxiaReach3AsyIdx = allAsyIdx[3,startAtaxia : startAtaxia +n,:]
statAtaxiaReach3SmoIdx = allSmoIdx[3,startAtaxia : startAtaxia +n,:]  

"REACH 5"
statNormalReach5LinIdx = allLinIdx[5,startAtaxia -n : startAtaxia,:]  
statNormalReach5AsyIdx = allAsyIdx[5,startAtaxia -n : startAtaxia,:]
statNormalReach5SmoIdx = allSmoIdx[5,startAtaxia -n : startAtaxia,:]  

statAtaxiaReach5LinIdx = allLinIdx[5,startAtaxia : startAtaxia +n,:]  
statAtaxiaReach5AsyIdx = allAsyIdx[5,startAtaxia : startAtaxia +n,:]
statAtaxiaReach5SmoIdx = allSmoIdx[5,startAtaxia : startAtaxia +n,:]





"PROCESS DATA"


"REACH 1"
"LIN IDX mean&std in normal and ataxia"
meanNormalReach1LinIdx = np.mean(statNormalReach1LinIdx, axis =0)
meanAtaxiaReach1LinIdx = np.mean(statAtaxiaReach1LinIdx, axis =0)

stdNormalReach1LinIdx  = np.std(statNormalReach1LinIdx, axis =0)
stdAtaxiaReach1LinIdx  = np.std(statAtaxiaReach1LinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalReach1AsyIdx = np.mean(statNormalReach1AsyIdx, axis =0)
meanAtaxiaReach1AsyIdx= np.mean(statAtaxiaReach1AsyIdx, axis =0)

stdNormalReach1AsyIdx  = np.std(statNormalReach1AsyIdx, axis =0)
stdAtaxiaReach1AsyIdx= np.std(statAtaxiaReach1AsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalReach1SmoIdx = np.mean(statNormalReach1SmoIdx, axis =0)
meanAtaxiaReach1SmoIdx= np.mean(statAtaxiaReach1SmoIdx, axis =0)

stdNormalReach1SmoIdx  = np.std(statNormalReach1SmoIdx, axis =0)
stdAtaxiaReach1SmoIdx= np.std(statAtaxiaReach1SmoIdx, axis =0)



"REACH 3"
"LIN IDX mean&std in normal and ataxia"
meanNormalReach3LinIdx = np.mean(statNormalReach3LinIdx, axis =0)
meanAtaxiaReach3LinIdx = np.mean(statAtaxiaReach3LinIdx, axis =0)

stdNormalReach3LinIdx  = np.std(statNormalReach3LinIdx, axis =0)
stdAtaxiaReach3LinIdx  = np.std(statAtaxiaReach3LinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalReach3AsyIdx = np.mean(statNormalReach3AsyIdx, axis =0)
meanAtaxiaReach3AsyIdx= np.mean(statAtaxiaReach3AsyIdx, axis =0)

stdNormalReach3AsyIdx  = np.std(statNormalReach3AsyIdx, axis =0)
stdAtaxiaReach3AsyIdx= np.std(statAtaxiaReach3AsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalReach3SmoIdx = np.mean(statNormalReach3SmoIdx, axis =0)
meanAtaxiaReach3SmoIdx= np.mean(statAtaxiaReach3SmoIdx, axis =0)

stdNormalReach3SmoIdx  = np.std(statNormalReach3SmoIdx, axis =0)
stdAtaxiaReach3SmoIdx= np.std(statAtaxiaReach3SmoIdx, axis =0)





"REACH 5"
"LIN IDX mean&std in normal and ataxia"
meanNormalReach5LinIdx = np.mean(statNormalReach5LinIdx, axis =0)
meanAtaxiaReach5LinIdx = np.mean(statAtaxiaReach5LinIdx, axis =0)

stdNormalReach5LinIdx  = np.std(statNormalReach5LinIdx, axis =0)
stdAtaxiaReach5LinIdx  = np.std(statAtaxiaReach5LinIdx, axis =0)

"ASY IDX mean&std in normal and ataxia"
meanNormalReach5AsyIdx = np.mean(statNormalReach5AsyIdx, axis =0)
meanAtaxiaReach5AsyIdx= np.mean(statAtaxiaReach5AsyIdx, axis =0)

stdNormalReach5AsyIdx  = np.std(statNormalReach5AsyIdx, axis =0)
stdAtaxiaReach5AsyIdx= np.std(statAtaxiaReach5AsyIdx, axis =0)

"SMO IDX mean&std in normal and ataxia"
meanNormalReach5SmoIdx = np.mean(statNormalReach5SmoIdx, axis =0)
meanAtaxiaReach5SmoIdx= np.mean(statAtaxiaReach5SmoIdx, axis =0)

stdNormalReach5SmoIdx  = np.std(statNormalReach5SmoIdx, axis =0)
stdAtaxiaReach5SmoIdx= np.std(statAtaxiaReach5SmoIdx, axis =0)














"INIT PLOTTING"
fig1   = plt.figure("Workspace", figsize=(20,10))
reach1 = plt.figtext(.20, 0.90, "REACH 1" , style='normal', bbox={'facecolor':'orangered'})
reach2 = plt.figtext(.48, 0.90, "REACH 3" , style='normal', bbox={'facecolor':'orangered'})
reach3 = plt.figtext(.76, 0.90, "REACH 5" , style='normal', bbox={'facecolor':'orangered'})

LinIdx = plt.figtext(.01, 0.77, "LIN IDX" , style='normal', bbox={'facecolor':'orangered'})
Asyidx = plt.figtext(.01, 0.49, "ASY IDX" , style='normal', bbox={'facecolor':'orangered'})
SmoIdx = plt.figtext(.01, 0.21, "SMO IDX" , style='normal', bbox={'facecolor':'orangered'})


"PLOT REACH 1 LINEARITY INDEX mean&std in normal and ataxia"
reach1LinIdxPlot = fig1.add_subplot(331)

reach1LinIdxPlot.set_ylim([0,5.0])
reach1LinIdxPlot.set_xlim([0,(maxSeed *2) +1])
reach1LinIdxPlot.set_yticks(np.arange(0, 5.0, 0.5))
reach1LinIdxPlot.set_xticks(np.arange(1, (maxSeed *2) +1, 1))

reach1LinIdxPlot.bar(np.linspace(1,maxSeed,maxSeed), meanNormalReach1LinIdx, width = 0.7, color ='blue', yerr = stdNormalReach1LinIdx)
reach1LinIdxPlot.bar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach1LinIdx, width = 0.7, color ='red', yerr = stdAtaxiaReach1LinIdx)

"PLOT REACH 3 LINEARITY INDEX mean&std in normal and ataxia"
reach3LinIdxPlot = fig1.add_subplot(332)

reach3LinIdxPlot.set_ylim([0,5.0])
reach3LinIdxPlot.set_xlim([0,(maxSeed *2) +1])
reach3LinIdxPlot.set_yticks(np.arange(0, 5.0, 0.5))
reach3LinIdxPlot.set_xticks(np.arange(1, (maxSeed *2) +1, 1))

reach3LinIdxPlot.bar(np.linspace(1,maxSeed,maxSeed), meanNormalReach3LinIdx, width = 0.7, color ='blue', yerr = stdNormalReach3LinIdx)
reach3LinIdxPlot.bar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach3LinIdx, width = 0.7, color ='red', yerr = stdAtaxiaReach3LinIdx)

"PLOT REACH 5 LINEARITY INDEX mean&std in normal and ataxia"
reach5LinIdxPlot = fig1.add_subplot(333)

reach5LinIdxPlot.set_ylim([0,5.0])
reach5LinIdxPlot.set_xlim([0,(maxSeed *2) +1])
reach5LinIdxPlot.set_yticks(np.arange(0, 5.0, 0.5))
reach5LinIdxPlot.set_xticks(np.arange(1, (maxSeed *2) +1, 1))

reach5LinIdxPlot.bar(np.linspace(1,maxSeed,maxSeed), meanNormalReach5LinIdx, width = 0.7, color ='blue', yerr = stdNormalReach5LinIdx)
reach5LinIdxPlot.bar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach5LinIdx, width = 0.7, color ='red', yerr = stdAtaxiaReach5LinIdx)




"PLOT REACH 1 ASYMMETRY INDEX mean&std in normal and ataxia"
reach1AsyIdxPlot = fig1.add_subplot(334)

reach1AsyIdxPlot.set_ylim([0,10.0])
reach1AsyIdxPlot.set_xlim([0,(maxSeed *2) +1])
reach1AsyIdxPlot.set_yticks(np.arange(0, 10.0, 1.0))
reach1AsyIdxPlot.set_xticks(np.arange(1, (maxSeed *2) +1, 1))

reach1AsyIdxPlot.bar(np.linspace(1,maxSeed,maxSeed), meanNormalReach1AsyIdx, width = 0.7, color ='blue', yerr = stdNormalReach1AsyIdx)
reach1AsyIdxPlot.bar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach1AsyIdx, width = 0.7, color ='red', yerr = stdAtaxiaReach1AsyIdx)

"PLOT REACH 3 ASYMMETRY INDEX mean&std in normal and ataxia"
reach3AsyIdxPlot = fig1.add_subplot(335)

reach3AsyIdxPlot.set_ylim([0,5.0])
reach3AsyIdxPlot.set_xlim([0,(maxSeed *2) +1])
reach3AsyIdxPlot.set_yticks(np.arange(0, 15.0, 1.))
reach3AsyIdxPlot.set_xticks(np.arange(1, (maxSeed *2) +1, 1))

reach3AsyIdxPlot.bar(np.linspace(1,maxSeed,maxSeed), meanNormalReach3AsyIdx, width = 0.7, color ='blue', yerr = stdNormalReach3AsyIdx)
reach3AsyIdxPlot.bar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach3AsyIdx, width = 0.7, color ='red', yerr = stdAtaxiaReach3AsyIdx)

"PLOT REACH 5 ASYMMETRY INDEX mean&std in normal and ataxia"
reach5AsyIdxPlot = fig1.add_subplot(336)

reach5AsyIdxPlot.set_ylim([0,5.0])
reach5AsyIdxPlot.set_xlim([0,(maxSeed *2) +1])
reach5AsyIdxPlot.set_yticks(np.arange(0, 15.0, 1.))
reach5AsyIdxPlot.set_xticks(np.arange(1, (maxSeed *2) +1, 1))

reach5AsyIdxPlot.bar(np.linspace(1,maxSeed,maxSeed), meanNormalReach5AsyIdx, width = 0.7, color ='blue', yerr = stdNormalReach5AsyIdx)
reach5AsyIdxPlot.bar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach5AsyIdx, width = 0.7, color ='red', yerr = stdAtaxiaReach5AsyIdx)





"PLOT REACH 1 SMOOTHNESS INDEX mean&std in normal and ataxia"
reach1SmoIdxPlot = fig1.add_subplot(337)

reach1SmoIdxPlot.set_ylim([0,15])
reach1SmoIdxPlot.set_xlim([0,(maxSeed *2) +1])
reach1SmoIdxPlot.set_yticks(np.arange(0, 15, 1))
reach1SmoIdxPlot.set_xticks(np.arange(1, (maxSeed *2) +1, 1))

reach1SmoIdxPlot.bar(np.linspace(1,maxSeed,maxSeed), meanNormalReach1SmoIdx, width = 0.7, color ='blue', yerr = stdNormalReach1SmoIdx)
reach1SmoIdxPlot.bar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach1SmoIdx, width = 0.7, color ='red', yerr = stdAtaxiaReach1SmoIdx)

"PLOT REACH 3 SMOOTHNESS INDEX mean&std in normal and ataxia"
reach3SmoIdxPlot = fig1.add_subplot(338)

reach3SmoIdxPlot.set_ylim([0,15])
reach3SmoIdxPlot.set_xlim([0,(maxSeed *2) +1])
reach3SmoIdxPlot.set_yticks(np.arange(0, 15, 1))
reach3SmoIdxPlot.set_xticks(np.arange(1, (maxSeed *2) +1, 1))

reach3SmoIdxPlot.bar(np.linspace(1,maxSeed,maxSeed), meanNormalReach3SmoIdx, width = 0.7, color ='blue', yerr = stdNormalReach3SmoIdx)
reach3SmoIdxPlot.bar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach3SmoIdx, width = 0.7, color ='red', yerr = stdAtaxiaReach3SmoIdx)

"PLOT REACH 5 SMOOTHNESS INDEX mean&std in normal and ataxia"
reach5SmoIdxPlot = fig1.add_subplot(339)

reach5SmoIdxPlot.set_ylim([0,15])
reach5SmoIdxPlot.set_xlim([0,(maxSeed *2) +1])
reach5SmoIdxPlot.set_yticks(np.arange(0, 15, 1))
reach5SmoIdxPlot.set_xticks(np.arange(1, (maxSeed *2) +1, 1))

reach5SmoIdxPlot.bar(np.linspace(1,maxSeed,maxSeed), meanNormalReach5SmoIdx, width = 0.7, color ='blue', yerr = stdNormalReach5SmoIdx)
reach5SmoIdxPlot.bar(np.linspace((maxSeed +1),maxSeed*2,maxSeed), meanAtaxiaReach5SmoIdx, width = 0.7, color ='red', yerr = stdAtaxiaReach5SmoIdx)


#ax11.bar(np.linspace(1,maxSeed*2,maxSeed*2), allMeanReach1LinIdx, yerr = allStdReach1LinIdx )


#gs = plt.GridSpec(11,11)

#
#
#

#title4 = plt.figtext(.01, 0.77, "LIN IDX" , style='normal', bbox={'facecolor':'orangered'})
#title5 = plt.figtext(.01, 0.49, "ASY IDX" , style='normal', bbox={'facecolor':'orangered'})
#title6 = plt.figtext(.01, 0.21, "SMO IDX" , style='normal', bbox={'facecolor':'orangered'})




"""

ax11 = fig1.add_subplot(gs[0:3, 0:3])





ax12 = fig1.add_subplot(gs[4:7, 0:3])


ax13 = fig1.add_subplot(gs[8:11, 0:3])





  



ax31 = fig1.add_subplot(gs[0:3, 4:7])
ax32 = fig1.add_subplot(gs[4:7, 4:7])
ax33 = fig1.add_subplot(gs[8:11, 4:7])
#title3 = plt.figtext(.5, 0.90, "LINEARITY INDEX REACH 3" , style='normal', bbox={'facecolor':'orangered'})

#ax3.bar(np.linspace(0,maxSeed*2,maxSeed*2), np.hstack([meanNormalReach3LinIdx,meanAtaxiaReach3LinIdx]))
#ax3.errorbar(np.linspace(0,maxSeed*2,maxSeed*2), np.hstack([stdNormalReach3LinIdx,stdAtaxiaReach3LinIdx]))
 



ax51 = fig1.add_subplot(gs[0:3, 8:11])
ax52 = fig1.add_subplot(gs[4:7, 8:11]) 
ax53 = fig1.add_subplot(gs[8:11, 8:11])   

#title5 = plt.figtext(.7, 0.90, "LINEARITY INDEX REACH 5" , style='normal', bbox={'facecolor':'orangered'})
        
#ax5.bar(np.linspace(0,maxSeed*2,maxSeed*2), np.hstack([meanNormalReach5LinIdx,meanAtaxiaReach5LinIdx]))
#ax5.errorbar(np.linspace(0,maxSeed*2,maxSeed*2), np.hstack([stdNormalReach5LinIdx,stdAtaxiaReach5LinIdx]))
"""





    
    
    



    
    
    
    



