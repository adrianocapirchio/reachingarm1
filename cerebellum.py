# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 00:09:21 2018

@author: Alex
"""

import numpy as np
import utilities as utils

class Cerebellum():
    
    def init(self, MULTINET, goalList, stateBg, maxStep, DOF = 2):
        
        
        
        self.dT = 0.95
        self.tau  = 1.
        self.C1 = self.dT / self.tau
        self.C2 = 1. - self.C1
        
        # LEARNING
        self.cbETA = 6.0 * 10 ** (-1) # 0.0025 SLOW # 0.025 fast
        
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
        
        self.tau1 = 1.0       
        # TEACHING 
    #    self.trainOut = np.ones(DOF) * 0.5
        self.errorOut = np.zeros(2)
    #                           
    #    self.desOutBuff = np.ones([DOF , maxStep]) * 0.5
        
       # self.trainEstVision = np.zeros(2)
      #  self.errorTrainEstVision = np.zeros(2)
      #  self.estVision = np.zeros(2)
      #  self.estVisionDistance =0
      #  self.errorEstVision = np.zeros(2)
     #   self.prvErrorEstVision = np.zeros(2)

        
        
     #   self.trainEstWrist = np.ones(len(np.hstack([wristState,wristState]))) * 0.5
     #   self.errorEstWrist = np.zeros(len(wristState))
        
        # OUTPUT
        self.currU = np.zeros(2)
        self.prvU = self.currU.copy()
        
        self.currI = np.ones(2) * 0.5
        self.prvI = self.currI.copy()
        
        self.damageI = np.ones(2) * 0.5
        
        self.trainU = np.zeros(2)
        self.prvTrainU = self.trainU.copy() 
        
        self.trainI = np.ones(2) * 0.5
        self.prvTrainI = self.trainI.copy()
        
        
      #  self.currOut = np.ones(DOF) * 0.5
      #  self.leakedOut = np.ones(DOF) * 0.5
      #  self.prvOut = np.ones(DOF) * 0.5
        
        
     #   self.estWrist = np.zeros(2)
                               
     #   self.trialFwdError = 0.
        
        
    def epochReset(self, maxStep,DOF = 2):
        
        self.currU = np.zeros(2)
        self.prvU = self.currU.copy()
        
        self.currI = np.ones(2) * 0.5
        self.prvI = self.currI.copy()
        
        self.damageI = np.ones(2) * 0.5
        
        self.trainU = np.zeros(2)
        self.prvTrainU = self.trainU.copy()
        
        self.trainI = np.ones(2) * 0.5
        self.prvTrainI = self.trainI.copy()
        
        self.errorOut = np.zeros(2)
        
        
        # OUTPUT
      #  self.currOut = np.ones(DOF) * 0.5
      #  self.leakedOut = np.ones(DOF) * 0.5
      #  self.prvOut = np.ones(DOF) * 0.5
                             
                             
      #  self.desOutBuff = np.ones([DOF , maxStep])* 0.5
        
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
      
    def trialReset(self, maxStep, DOF = 2):
        
        self.trainU = np.zeros(2)
        self.prvTrainU = self.trainU.copy()
        
        self.trainI = np.ones(2) * 0.5
        self.prvTrainI = self.trainI.copy()
        
        self.errorOut = np.zeros(2)
        
        
      #  self.desOutBuff = np.ones([DOF , maxStep])* 0.5  
      
    
    def compU(self, state):
        self.currU = self.C2 * self.prvU + self.C1 * np.dot(self.w.T, state)
        
    def spreading(self):
        self.currI = utils.sigmoid(self.currU)

    
#    def spreading(self,state):
#        self.currOut = utils.sigmoid(np.dot(self.w.T, state))
        
    def trainCb(self,state, gangliaI, rew):
        self.prvTrainU = self.trainU.copy()
        self.trainU = self.C2 * self.prvTrainU + self.C1 * np.dot(self.w.T, state)
        self.trainI = utils.sigmoid(self.trainU)
        self.errorOut = gangliaI - self.trainI
        self.w +=  self.cbETA * np.outer(state, self.errorOut) * self.trainI * (1. - self.trainI)