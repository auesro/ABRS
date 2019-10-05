#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:02:58 2019

@author: auesro
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import platform
from scipy import ndimage
from scipy import misc


#import ABRS_data_vis

dirPathInput = '/home/auesro/Desktop/Store/hour1_fb1'

fl=0 #select a file number in the ST3C folder

fileList = sorted(os.listdir(dirPathInput))
inputFileName = fileList[fl];

OSplatform = platform.system()

if OSplatform == 'Linux':
    fileDirPathInputName = dirPathInput + '/' + inputFileName
if OSplatform == 'Windows':
    fileDirPathInputName = dirPathInput + '\\' + inputFileName
if OSplatform == 'Darwin':
    fileDirPathInputName = dirPathInput + '/' + inputFileName
    
with open(fileDirPathInputName, "rb") as f:

    dict3C = pickle.load(f)
    

recIm3C = dict3C["recIm3C"]    
behPredictionRec = dict3C["behPredictionRec"]
    
print(fileDirPathInputName)

#Plot all images in recIm3C
fig=plt.figure(figsize=(10, 10))

imDiff = np.zeros((80,80))

for i in range(1, len(recIm3C)):
        fig.add_subplot(8, 8, i)
        
        im3C = recIm3C[i,:,:,:]/255
        imDiff = imDiff + im3C[...,1]
        
        imToPlot = recIm3C[i,:,:,:]/255            
        
        plt.imshow(imToPlot)
        #plt.title(str(behPredictionRec[i])) #Plots labels when they exist
                
plt.show()


#Plot single image i from recIm3C
#i=1
#plt.imshow(recIm3C[i,:,:,:]/255);plt.show()

