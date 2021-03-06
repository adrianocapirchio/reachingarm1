# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 21:07:36 2018

@author: Alex
"""

import numpy as np
import utilities as utils

class armReaching6targets:
    
    def init(self, maxStep, maxEpoch):
        
        
        
        self.goalIdx = 0
        
        
        
        
        
        
        
                            
        self.goalList = [np.array([-0.07 -.05, 0.16 +0.04]),
                         np.array([ 0.10 -.05, 0.16 +0.04]),
                         np.array([-0.24 -.05, 0.16 +0.04]),
                         np.array([-0.07 -.05, 0.42 +0.04]),        
                         np.array([ 0.10 -.05, 0.42 +0.04]),
                         np.array([-0.24 -.05, 0.42 +0.04])]
        
        self.maxTrial = 7#len(self.goalList)
        
        
        
        self.goalPos = self.goalList[0].copy()
        self.prvGoalPos = self.goalPos.copy()
        
        self.currPos = np.zeros(2)
        self.prvPos = np.zeros(2)
        
        self.currVel = np.zeros(1)
        self.prvVel = np.zeros(1)
        self.currAcc = np.zeros(1)
        self.prvAcc = np.zeros(1)
        self.currJerk = np.zeros(1)
        
        self.goalPositionHistory = np.zeros([2, self.maxTrial, maxEpoch])

        
        self.trialArmAngles = np.zeros([2, maxStep, self.maxTrial, maxEpoch])
        self.goalAnglesHistory = np.zeros([2, maxStep, self.maxTrial, maxEpoch])
        
        self.trialGangliaAngles = np.zeros([2, maxStep, self.maxTrial, maxEpoch])
        self.trialCerebAngles = np.zeros([2, maxStep, self.maxTrial, maxEpoch])
        self.trialmtxAngles = np.zeros([2, maxStep, self.maxTrial, maxEpoch])
        
        self.trialTrajectories = np.zeros([2, maxStep, self.maxTrial, maxEpoch])
        self.trialVelocity = np.zeros([maxStep, self.maxTrial, maxEpoch])
        self.trialAccelleration = np.zeros([maxStep, self.maxTrial, maxEpoch])
        self.trialJerk = np.zeros([maxStep, self.maxTrial, maxEpoch])       
        
        
    def epochReset(self):
        
        self.currPos = np.zeros(2)
        self.prvPos = np.zeros(2)
        
        self.currVel = np.zeros(1)
        self.prvVel = np.zeros(1)
        self.currAcc = np.zeros(1)
        self.prvAcc = np.zeros(1)
        self.currJerk = np.zeros(1)
        
        
        
        
        
        

    
    
    
    
    
    
    
    def setGoal(self, trial):
        
        if trial%2 == 0:
            self.goalPos = self.goalList[0].copy()       
        elif trial == 1:
            self.goalPos = self.goalList[5].copy()
        elif trial == 3:
            self.goalPos = self.goalList[4].copy()
        elif trial == 5:
            self.goalPos = self.goalList[3].copy()
        
        
     ##   if trial%2 == 0:
     #       self.goalPos = self.goalList[0].copy()            
     #   elif trial == 1:
     #       self.goalPos = self.goalList[5].copy()
     #   elif trial == 3:
     #       self.goalPos = self.goalList[3].copy()
     #   elif trial == 5:
     #       self.goalPos = self.goalList[4].copy()
            
        
        
        
  #      if reachType== 0:
            
       #     if trial == 0:
       #         self.goalPos = self.goalList[0].copy()
       #     elif trial == 1:
   #        self.goalPos = self.goalList[5].copy()
       #     elif trial == 2:
        #        self.goalPos = self.goalList[0].copy()    
            
    #    if reachType== 1:
            
  #          if trial == 0:
  #              self.goalPos = self.goalList[0].copy()
  #          elif trial == 1:
     #       self.goalPos = self.goalList[3].copy()
 #           elif trial == 2:
  #              self.goalPos = self.goalList[0].copy()  
                
     #   if reachType== 2:
            
   #         if trial == 0:
    #            self.goalPos = self.goalList[0].copy()
     #       elif trial == 1:
      #      self.goalPos = self.goalList[4].copy()
         #   elif trial == 2:
         #       self.goalPos = self.goalList[0].copy()  
                
                
    


            
    def computeDistance(self):
        self.distance = utils.distance(self.currPos,self.goalPos)   
        
        
    
    
    "set goal to a random position in goal list once for epoch"        
    def randomGoal(self,trial):
        
        if trial == 0:
            self.tempGoalList = self.goalList[0:8]
    
        i = np.random.randint(len(self.tempGoalList))
        self.goalPos = self.tempGoalList[i]
        del self.tempGoalList[i]
        
    
    
    
    
    
    def goalIndex0(self, trial):
        
        for i in xrange(self.maxTrial):
            if trial == i:
                self.goalIdx = i
        
    #    if trial == 0:
    #        self.goalIdx = 0
     #   if trial == 1:
     #       self.goalIdx = 1
      #  if trial == 2:
      #      self.goalIdx = 2
      #  if trial == 3:
      #      self.goalIdx = 3
      #  if trial == 4:
      #      self.goalIdx = 4
      #  if trial == 5:
      #      self.goalIdx = 5
      #  if trial == 6:
      #      self.goalIdx = 6
            
        
      #  for i in xrange(len(self.goalList)):
      #      if (self.goalPos == self.goalList[i]).all():
      #      #    print i
      #          self.goalIdx = i
                
    







            
    def goalIndex1(self, trial):
        
        
        if trial%2 == 0:
            self.goalIdx = 0
        elif trial == 1:
            self.goalIdx = 1
        elif trial == 3:
            self.goalIdx = 2
        elif trial == 5:
            self.goalIdx = 3
  #☻#      elif trial == 2:
     #       if reachType == 0:
     #           self.goalIdx = 3
     #       elif reachType == 1:
     #           self.goalIdx = 4
     #       elif reachType == 2:
     #           self.goalIdx = 5
            
  
                
                
                
                
                
                
            