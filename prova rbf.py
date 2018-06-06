# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:50:41 2018

@author: Alex
"""

import utilities as utils
import numpy as np
import scipy as scy
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt


grid = utils.build2DGrid(10,0,1).reshape(2,100)
activation = np.zeros(len(grid[0]))
sig = (1. / (10 * 2))
position = np.random.uniform(0.0,1.0,2)

for i in xrange(len(grid[0])):
    activation[i] = np.exp(-utils.distance(position,grid[:,i])**2  / (2 * sig ** 2.))
