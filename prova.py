# -*- coding: utf-8 -*-
"""
Created on Fri Jun 01 15:48:27 2018

@author: Alex
"""

import matplotlib.pyplot as plt


fig1   = plt.figure("FORWARD REACHING INDEX", figsize=(20,10))
reach1 = plt.figtext(.20, 0.90, "REACH 1" , style='normal', bbox={'facecolor':'orangered'})

fig2   = plt.figure("BACKWARD REACHING INDEX", figsize=(20,10))
reach2 = plt.figtext(.20, 0.90, "REACH 2" , style='normal', bbox={'facecolor':'orangered'})
