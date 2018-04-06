# -*- coding: utf-8 -*-
"""
Created on Fri Mar 09 23:00:04 2018

@author: Alex
"""

import numpy as np
import utilities as utils



class actorCritic:
    
    def init(self, MULTINET, VISION, GOAL_VISION , AGENT_VISION, PROPRIOCEPTION, ELBOW_PROPRIOCEPTION, SHOULDER_PROPRIOCEPTION, maxStep, maxTrial, goalList, perfBuff, cerebellumTime, DOF = 2):

        self.noiseMag0 = 1.2
        self.noiseMag1 = 1.2 #0.16
        self.DELTM = 0.04 #0.56
        self.TAU = 1.
        
        # LEARNING PARAM2ETERS
        self.ACT_ETA1 = 0.2 * 10 ** (   -1)#☻ slow 0.1
        self.ACT_ETA2 = 1.0 * 10 ** (   -6)# # slow 0.00025
        self.CRIT_ETA = 1.0 * 10 ** (   -4) #0.00000000001# slow 0.01 -8
        
        self.DISC_FACT = 0.99

        
        self.bias = np.ones(1)
        self.currState = np.array([])
        
    #    if PROPRIOCEPTION == True:
    #        self.proprioceptionState = np.array([])
            
   #         self.proprioceptionInputUnits = 31 **2
    #        self.intervalsProprioception = int(np.sqrt(self.proprioceptionInputUnits)) -1
    #        self.sigmaProprioception = (1. / (self.intervalsProprioception* 2))
    #        self.proprioceptionGrid = utils.build2DGrid(self.intervalsProprioception +1, 0., 1.)
    #        self.proprioceptionRawState = np.zeros([self.intervalsProprioception +1,self.intervalsProprioception +1])
    #        self.proprioceptionState = np.zeros(self.proprioceptionInputUnits)
            
     #       self.currState = np.zeros(len(self.currState)+len(self.proprioceptionState))
        
        
        if PROPRIOCEPTION == True:
            
            self.proprioceptionState = np.array([])
            
            if ELBOW_PROPRIOCEPTION == True:
                
                self.elbowInputUnits= 2501# 901 fast #2101 slow#2501
                self.intervalsElbow = self.elbowInputUnits - 1
                self.sigElbow = 1. / (self.intervalsElbow * 2)
                self.elbowGrid = np.linspace(0, 1, self.elbowInputUnits)
                self.elbowState= np.zeros(self.elbowInputUnits)
                self.proprioceptionState = np.zeros(len(self.proprioceptionState) + len(self.elbowState))
                
            if SHOULDER_PROPRIOCEPTION == True:
                
                self.shoulderInputUnits= 2501 #1501 fast #2501 slow 
                self.intervalsShoulder =  self.shoulderInputUnits -1
                self.sigShoulder = 1. / (self.intervalsShoulder * 2)
                self.shoulderGrid = np.linspace(0, 1, self.shoulderInputUnits)
                self.shoulderState= np.zeros(self.shoulderInputUnits)
                self.proprioceptionState = np.zeros(len(self.proprioceptionState) + len(self.shoulderState))
                
            self.currState = np.zeros(len(self.currState)+len(self.proprioceptionState))
        
        
        if VISION == True:
            
            self.visionState = np.array([])
            
            if GOAL_VISION == True:
                self.goalVisionInputUnits = 21**2 #201
                self.goalVisionIntervals = int(np.sqrt(self.goalVisionInputUnits)) -1               
                self.goalVisionSig= (1. / (self.goalVisionIntervals* 2))
                self.goalVisionGrid = utils.build2DGrid(self.goalVisionIntervals +1, 0., 1.)
                self.goalVisionRawState = np.zeros([self.goalVisionIntervals +1,self.goalVisionIntervals +1])
                self.goalVisionState = np.zeros(self.goalVisionInputUnits)
                self.visionState = np.zeros(len(self.visionState) + len(self.goalVisionState))
                
            if AGENT_VISION == True:
                self.agentVisionInputUnits = 20**2
                self.agentVisionIntervals = int(np.sqrt(self.agentVisionInputUnits)) - 1               
                self.agentVisionSig= 1. / (self.agentVisionIntervals* 2)
                self.agentVisionGrid = utils.build2DGrid(self.agentVisionIntervals +1, 0., 1.)
                self.agentVisionRawState = np.zeros([self.agentVisionIntervals +1,self.agentVisionIntervals +1])
                self.agentVisionState = np.zeros(self.agentVisionInputUnits)
                self.visionState = np.zeros(len(self.visionState) + len(self.agentVisionState))
            
            self.currState = np.zeros(len(self.currState)+len(self.visionState))   
            
        
        # STATE PARAMETERS
        self.prvState = self.currState.copy()
        self.prv5State = self.currState.copy()
        self.biasedState = np.hstack([self.bias,self.currState])
        self.prvBiasedState =self.biasedState.copy()
        self.prv5BiasedState =self.biasedState.copy()
        
        self.actW = np.zeros([len(self.biasedState), DOF])
        self.critW= np.zeros(len(self.currState))
        
        if MULTINET == True:
            self.multiActW = np.zeros([len(self.biasedState), DOF, len(goalList)])         
            self.multiCritW = np.zeros([len(self.currState), len(goalList)])
        
        
        self.rewardCounter = np.zeros(len(goalList))

        self.actRew = 0
        self.surp = np.zeros(1)
        self.currCritOut = np.zeros(1)
        self.prvCritOut = np.zeros(1)
        
        self.currActOut = np.ones(DOF)*0.5
        self.prvActOut = np.ones(DOF)*0.5
        self.trainOut = np.ones(DOF)*0.5
        self.prvTrainOut = np.ones(DOF)*0.5
        self.currNoise = np.zeros(2) #np.array([0.42586088, 0.79620307])
        self.prvNoise = self.currNoise.copy()
     
        self.performance = np.zeros([perfBuff, 6])
        
        self.stateBuff = np.zeros([len(self.biasedState) , maxStep / cerebellumTime])
        self.desOutBuff = np.ones([DOF, maxStep / cerebellumTime]) * 0.5
        
        
        
        
    def epochReset(self, maxStep, cerebellumTime, DOF = 2):
        
        # STATE PARAMETERS
        self.currState *= 0
        self.prvState = self.currState.copy()
        self.prv5State = self.currState.copy()
        self.biasedState = np.hstack([self.bias,self.currState])
        self.prvBiasedState = np.hstack([self.bias,self.currState])
        self.prv5BiasedState = np.hstack([self.bias,self.currState])
        
        self.stateBuff = np.zeros([len(self.biasedState) , maxStep / cerebellumTime])
        self.desOutBuff = np.ones([DOF, maxStep / cerebellumTime]) * 0.5
        
        
        
        #◘ CRITIC PARAMETERS

        self.actRew = 0
        self.surp *= 0 
        self.currCritOut *= 0
        self.prvCritOut *= 0

        # ACTOR PARAMETERS
        self.currActOut = np.ones(DOF)*0.5
        self.prvActOut = np.ones(DOF)*0.5
        self.trainOut = np.ones(DOF)*0.5 
        self.prvTrainOut = np.ones(DOF)*0.5
        self.currNoise = np.zeros(2) #np.array([0.42586088, 0.79620307])
        self.prvNoise = self.currNoise.copy()  
    


    
    def trialReset(self, maxStep, cerebellumTime, DOF = 2):
        
        self.actRew = 0
        
        self.stateBuff = np.zeros([len(self.biasedState) , maxStep/ cerebellumTime])
        self.desOutBuff = np.ones([DOF, maxStep/ cerebellumTime]) * 0.5
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def acquireGoalVision(self, goalPosition):
        self.goalVisionRawState = utils.gaussian2D(goalPosition,self.goalVisionGrid,self.goalVisionSig,self.goalVisionRawState,self.goalVisionIntervals)
        self.goalVisionState = self.goalVisionRawState.ravel()
        
    def acquireAgentVision(self, agentPosition):
        self.agentVisionRawState = utils.gaussian2D(agentPosition,self.agentVisionGrid,self.agentVisionSig,self.agentVisionRawState,self.agentVisionIntervals)
        self.agentVisionState = self.agentVisionRawState.ravel()
        
    def acquireProprioception(self, armAngles):
        self.proprioceptionRawState = utils.gaussian2D(armAngles,self.proprioceptionGrid,self.sigmaProprioception,self.proprioceptionRawState,self.intervalsProprioception)
        self.proprioceptionState = self.proprioceptionRawState.ravel()
    
    
    def acquireElbowState(self, elbowPosition):
        self.elbowState = utils.gaussian(elbowPosition,self.elbowGrid,self.sigElbow)
    
    def acquireShoulderState(self, shoulderPosition):
        self.shoulderState = utils.gaussian(shoulderPosition,self.shoulderGrid,self.sigShoulder)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def noise(self,T): 
        self.C1 = self.DELTM/ self.TAU
        self.C2 = 1. - self.C1
     #   self.currNoise[0] = utils.limitRange((self.C2 * self.prvNoise[0] + self.C1 *  np.random.normal(0.5, self.noiseMag0, 1)), 0., 1.)
     #   self.currNoise[1] = utils.limitRange((self.C2 * self.prvNoise[1] + self.C1 *  np.random.normal(0.5, self.noiseMag1, 1)), 0., 1.)
      #  self.currNoise[1] = utils.limitRange((self.C2 * self.prvNoise[1] + (self.C1 *  np.random.normal(-self.currActOut[1] + 0.5, self.noiseMag1, 1))) * T, -1., 1.)
        self.currNoise = utils.limitRange((self.C2 * self.prvNoise + self.C1 * np.random.normal(0.0, self.noiseMag0, 2)) * T, -0.6, 0.6)   
                         
    def spreadAct(self):
        self.currActOut = utils.sigmoid(np.dot(self.actW.T, self.biasedState))
                         
    def spreadCrit(self): 
        self.currCritOut = np.dot(self.critW, self.currState)
        
    def compSurprise(self):
        self.surp = self.actRew + (self.DISC_FACT * self.currCritOut) - self.prvCritOut     
                                  
                                  
                                  
                                  
                                  
                                  
    
    
    
    
    def trainCrit(self):
        self.critW += self.CRIT_ETA * self.surp * self.prv5State
        
    def trainAct(self, prvNoisedOut):    
        self.actW += self.surp * self.ACT_ETA1 * np.outer(self.prv5BiasedState, self.prvNoise)
                               
    def trainAct2(self, cerebPrvOut):
        self.actW += self.ACT_ETA2 * np.outer(self.prvState, (cerebPrvOut- self.prvTrainOut))
                         
    
    
            
            
            
            
        