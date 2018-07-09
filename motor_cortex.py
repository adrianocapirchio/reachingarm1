# -*- coding: utf-8 -*-
"""
Created on Fri Jul 06 18:28:19 2018

@author: Alex
"""

import numpy as np
import utilities as utils

class Motor_cortex():
    
    def init(self, goal_state, DOF = 2):
        
        
        
        self.dT = 0.99
        self.tau  = 1.
        self.C1 = self.dT / self.tau
        self.C2 = 1. - self.C1
        
        # LEARNING
        self.ETA = 5.0 * 10 ** (-1)
        
        self.currState = np.zeros(len(goal_state))
        self.w = np.zeros([len(goal_state), DOF])
        
        self.errorOut = np.zeros(2)
        self.currU = np.zeros(2)
        self.currI = np.ones(2) * 0.5
                            
    def epochReset(self, DOF = 2):
        
        self.currU = np.zeros(2)
        self.currI = np.ones(2) * 0.5
                            
    def compU(self, state):
        self.currU = self.C2 * self.currU + self.C1 * np.dot(self.w.T, state)
    
    def compI(self):
        self.currI = utils.sigmoid(self.currU)
        
    
    def training(self,state, netout, r):
        self.errorOut = netout - self.currI
        self.w += r * self.ETA * np.outer(state, self.errorOut) * self.currI * (1.0 -self.currI)
        