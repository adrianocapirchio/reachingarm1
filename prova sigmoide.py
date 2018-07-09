# -*- coding: utf-8 -*-
"""
Created on Tue Jul 03 15:43:12 2018

@author: Alex
"""


import numpy as np
import matplotlib.pyplot as plt
import utilities as utils


x = np.linspace(-10.0, +10.0 , 1000)
y = utils.sigmoid(x)


fig1   = plt.figure("sigmoid", figsize=(8,8))
ax1    = fig1.add_subplot(111, aspect='equal')
ax1.set_xlim([-10.0,10.0])
ax1.set_ylim([-0.2,1.2])

sigmoid = ax1.scatter(x, y, color='blue', s = 1.0)






perfBuff = 100

fig2   = plt.figure("noise T decreasing", figsize=(8,8))
ax2    = fig2.add_subplot(111)
ax2.set_xlim([+0.0,+100.0])
ax2.set_ylim([-0.0,1.])


x1 = np.linspace( +0.0, +100.0, perfBuff)
y1 = np.zeros(len(x1))
y1 = 1. * utils.clipped_exp(-x1/perfBuff * 10.0)
y2 = np.zeros(len(x1))
y2 = 1.0 - x1/100

T = ax2.scatter(x1, y1, color='blue', s = 10.0)
T2 = ax2.scatter(x1, y2, color='blue', s = 10.0)


