# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:09:02 2018

@author: Alex
"""

import numpy as np
import scipy as scy


def changeRange(old_value, old_min, old_max, new_min, new_max):
    return (((old_value - old_min) * (new_max - new_min)) / (old_max - old_min)) + new_min

def limitRange(x, x_min, x_high):
    return np.maximum(x_min, np.minimum(x_high,x))


def gaussian(x, mu, sig):
    return np.exp(-((x - mu)**2/ (2 * sig ** 2.)))


def gaussian2D(curr2DPos, mu, sig, inputArray, intervals):
    for j in xrange(intervals +1):
        for i in xrange(intervals +1):
            inputArray[i,j] =  np.exp(- (np.sum((curr2DPos -mu[:,i,j])**2) / (2 * sig ** 2.)))       
    return inputArray

def gaussian3D(currAngles, mu, sig, wristRawPosState, interN):
    
    for k in xrange(interN +1):
        for j in xrange(interN +1):
            for i in xrange(interN +1):
                wristRawPosState[i,j,k] = np.exp(- np.sum(sqrDis(currAngles,mu[:,i,j,k]))  / (2 * sig ** 2.))
    return wristRawPosState








def clipped_exp(x):
    cx =np.clip(x, -700, 700)
    return np.exp(cx)

def sigmoid(x):
    return 1 / (1.0 + clipped_exp(-(2.0*x)))







def sqrDis(x1,x2):
    return (x1 - x2)**2

def distance(a,b):
    return np.sqrt(np.sum(sqrDis(a,b)))

















def build3DGrid(intervals, rangeMin, rangeMax):
    
    x = np.linspace(rangeMin[0],rangeMax[0],intervals)
    y = np.linspace(rangeMin[1],rangeMax[1],intervals)
    z = np.linspace(rangeMin[2],rangeMax[2],intervals)

    xxx, yyy, zzz = np.meshgrid(x,y,z)

    end = np.zeros([3,intervals,intervals,intervals])    
    end[0,:,:,:] = xxx
    end[1,:,:,:] = yyy
    end[2,:,:,:] = zzz
        
    return end

def build2DGrid(intervals, rangeMin, rangeMax):
    x = np.linspace(rangeMin, rangeMax ,intervals)
    y = np.linspace(rangeMin, rangeMax ,intervals)
    xx, yy = np.meshgrid(x, y)
    
    end = np.zeros([2,intervals,intervals])  
    end[0,:,:] = xx
    end[1,:,:] = yy
        
    return end







def rotation2D(coord,angle):   
    rad = np.deg2rad(angle)
    rotCoords = np.zeros(2)

    rotCoords[0] = coord[0] * np.cos(rad) - coord[1] * np.sin(rad)
    rotCoords[1] = coord[0] * np.sin(rad) + coord[1] * np.cos(rad)
    
    return rotCoords


def skew(vec):    
    hat = np.zeros([len(vec),len(vec)])  
    hat[0,1] = -vec[2]
    hat[0,2] = vec[1]
    hat[1,0] = vec[2]
    hat[1,2] = -vec[0]
    hat[2,0] = -vec[1]
    hat[2,1] = vec[0]
    return hat

def rotationMatrix(currAngles):
       
    ex = np.array([1,0,0])
    exHat = skew(ex)
    
    ey = np.array([0,1,0])
    eyHat = skew(ey)
    
    ez = np.array([0,0,1])
    ezHat = skew(ez)
    
    return np.dot(np.dot(scy.linalg.expm(-exHat * currAngles[0]), scy.linalg.expm(ezHat * currAngles[2])),scy.linalg.expm(eyHat * currAngles[1]))

def rotationVector(R):
    
    theta = np.arccos((np.trace(R)- 1)/2)
    
  #  print theta
    
    r = (theta/ (2*np.sin(theta))) * np.array([ R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1] ])
    
    return r












def linearityIndex(path, endPos, goalRange, prvGoalPos):
    
    pathx = np.trim_zeros(path[0,:], 'b')
    pathy = np.trim_zeros(path[1,:], 'b')
    
    trimmedPath = np.vstack([pathx, pathy])
    
  #  path = np.trim_zeros(path, 'b')
    
    pathLen = 0.
    
    startPos = path[:,0]
    
    minDistance = distance(startPos, endPos)
    
#   print path.shape

   # print distance(path[:, 0], path[:,1])
    
    for t in xrange(len(trimmedPath[0,:])):
        if t > 0:
       #     print i
        #    print distance(path[i -1], path[i])
            pathLen += distance(trimmedPath[:, t], trimmedPath[:,t -1])
        #    print pathLen
            
 #   print pathLen / minDistance
    
    return (pathLen + goalRange) / minDistance

def smoothnessIndex(jerkPath):
    
    jerkPath = np.trim_zeros(jerkPath, 'b')
    
    J2 = np.linalg.norm(jerkPath)**2
    J2int = np.sum(J2)
    logJ2int = np.log(np.sqrt(J2int))
    
    return logJ2int

def asimmetryIndex(velPath):
    
    velPath = np.trim_zeros(velPath, 'b')
    
    prePeak = np.argmax(velPath)
    print prePeak
    
    postPeak = len(velPath[prePeak:])
    
    if prePeak != 0:
        return float(postPeak)/float(prePeak)
    
    
    
    
    
    
    
    
    
    
    