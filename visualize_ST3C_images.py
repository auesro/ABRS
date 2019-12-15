#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Primoz Ravbar UCSB
# Licensed under BSD 2-Clause [see LICENSE for details]
# Written by Primoz Ravbar

"""
Modified on Sat Oct  5 10:02:58 2019
@author: Augusto Escalante
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import platform


# dirPathInput = '/home/augustoer/ABRS/hour120_Arena1' #lab path
dirPathInput = '/home/auesro/Desktop/ABRS/Test/Result' #home path
# dirPathInput = '/home/auesro/Desktop/ABRS/20795_Arena1' #home path logitech

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
   


recIm3C = dict3C#["recIm3C"]    
# behPredictionRec = dict3C["behPredictionRec"]
    
print(fileDirPathInputName)

#Plot all images in recIm3C
fig=plt.figure(figsize=(10, 10), dpi=300)

# imDiff = np.zeros((80,80)) #AER: Unknown function

for i in range(1, len(recIm3C)):
        fig.add_subplot(8, 8, i)
        
        # im3C = recIm3C[i,:,:,:]/255 #AER: Unknown function
        # imDiff = imDiff + im3C[...,1] #AER: Unknown function
        
        imToPlot = recIm3C[i,:,:,:]/255    
        
        plt.imshow(imToPlot)
        #plt.title(str(behPredictionRec[i])) #AER:Plots labels when they exist
                
plt.show()


# # Plot single image i from recIm3C
# i=1
# plt.imshow(recIm3C[i,:,:,:]/255);plt.show()

