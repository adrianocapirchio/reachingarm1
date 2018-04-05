# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 00:09:21 2018

@author: Alex
"""

import numpy as np
import utilities as utils

class Cerebellum():
    
    def init(self, MULTINET, goalList, stateBg, DOF = 2):
        
        
        
        # LEARNING
        self.cbETA = 0.2 * 10 ** (-1)# 0.0025 SLOW # 0.025 fast
        
        # STATE
        self.currState = np.zeros(len(stateBg))
   #     self.fwdVisionState = np.ones(len(np.hstack([visionState,wristState])))
   #     self.prvFwdVisionState = self.fwdVisionState.copy()
        
        # WEIGHTS
        self.w = np.zeros([len(stateBg), DOF])
        
        if MULTINET == True:
            self.multiCerebW = np.zeros([len(stateBg), DOF, len(goalList)])
 #       self.fwdVisionW = np.zeros([len(np.hstack([visionState,wristState])), 2])
  #      self.fwdWristW = np.zeros([len(np.hstack([wristState,wristState])), 3])
        
               
        # TEACHING 
        self.trainOut = np.ones(DOF) * 0.5
        self.errorOut = np.ones(DOF) * 0.5
        
       # self.trainEstVision = np.zeros(2)
      #  self.errorTrainEstVision = np.zeros(2)
      #  self.estVision = np.zeros(2)
      #  self.estVisionDistance =0
      #  self.errorEstVision = np.zeros(2)
     #   self.prvErrorEstVision = np.zeros(2)

        
        
     #   self.trainEstWrist = np.ones(len(np.hstack([wristState,wristState]))) * 0.5
     #   self.errorEstWrist = np.zeros(len(wristState))
        
        # OUTPUT
        self.currOut = np.ones(DOF) * 0.5
        self.prvOut = np.ones(DOF) * 0.5
        
        
     #   self.estWrist = np.zeros(2)
                               
     #   self.trialFwdError = 0.
        
        
    def epochReset(self, DOF = 2):
        
        # OUTPUT
        self.currOut = np.ones(DOF) * 0.5
        self.prvOut = np.ones(DOF) * 0.5
        
      #  self.estVision *= 0.
      #  self.estVisionDistance *=0
      #  self.errorEstVision *= 0
      #  self.prvErrorEstVision *= 0
        
 #   def trialReset(self):
        
        # OUTPUT
      #  self.currOut = np.ones(3) * 0.5
       # self.prvOut = np.ones(3) * 0.5
        
      #  self.trialFwdError *= 0 
      #  self.estVision *= 0.  
      #  self.estVisionDistance *=0
      #  self.errorEstVision *= 0 
      #  self.prvErrorEstVision *= 0       
      
        
    def spreading(self,state):
        self.currOut = utils.sigmoid(np.dot(self.w.T, state))
        
    def trainCb(self,state, ep):
        self.trainOut = utils.sigmoid(np.dot(self.w.T, state))
        self.errorOut = ep - self.trainOut
        self.w +=  self.cbETA * np.outer(state, self.errorOut)