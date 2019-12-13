#!/usr/bin/env python
# coding: utf-8

#  Copyright (c) 2019 Primoz Ravbar UCSB
#  Licensed under BSD 2-Clause [see LICENSE for details]
#  Written by Primoz Ravbar

"""
Modified on Thu Dec  12 12:04:58 2019
@author: Augusto Escalante
"""


# load ST-images and labels for the ConvNet training

import numpy as np
import scipy
from scipy import ndimage
from scipy import misc
import pickle
import pandas as pd
import time
import matplotlib.pyplot as plt
import cv2
import os
import matplotlib.colors as mcolors
import natsort
from PIL import Image

from sklearn.utils import shuffle

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from ABRS_modules import discrete_radon_transform
from ABRS_modules import etho2ethoAP
from ABRS_modules import smooth_1d
from ABRS_modules import create_LDA_training_dataset
from ABRS_modules import removeZeroLabelsFromTrainingData
from ABRS_modules import computeSpeedFromPosXY 

###############################################################
#Parameters to set by the user
###############################################################

#PATHS:
pathToABRSfolder = 'INSERT PATH TO ABRS MAIN FOLDER HERE'
# pathToABRSfolder = 'C:\\Users\\primo\\Desktop\\USB\\ABRS\\ABRS_Python_GHws1'

dirPathInput = 'INSERT PATH TO ABRS ST-images HERE';
fileList = natsort.natsorted(os.listdir(dirPathInput));
dirPathLabel = pathToABRSfolder + '\\Labels';

idxLabelDirPathFileName = dirPathLabel + '\\' + 'idxRecLabelLiAVI_manual_scoring'; #path to label file
#idxLabelDirPathFileName = dirPathLabel + '\\' + 'labelCS1fb1_SS';

outputFolderEtho = pathToABRSfolder + '\\Etho';

#PARAMETERS:
labelShift = 10; # label onset correction
thresholdMovement=250; #this is min. signal threshold (frames with no movement will not be used in training)

##################################################################

with open(idxLabelDirPathFileName, "rb") as f:
     idxLabel = pickle.load(f)



shL = np.shape(idxLabel);
labelShftRight = np.hstack((np.zeros((1,labelShift)),idxLabel[:,0:shL[1]-labelShift])); # works with janelia data 11/16/2018 # shift 15 works too
idxLabel = labelShftRight;  
idxLabel[idxLabel==0]=7

numbFiles = np.shape(fileList)[0] #
skipFilesNumb =1;
skipFrameNumb=1;

normalizeByMax = 1;



yi = np.zeros((1,10))
yiVect = np.zeros((1,1))

Make this more standard:
rtImRec = np.zeros((50000,80,80,3))

indIm = 0

for fl in range(0, numbFiles-1, skipFilesNumb): #

    inputFileName = fileList[fl];

    fileDirPathInputName = dirPathInput + '\\' + inputFileName
    
    print(fileDirPathInputName)

    with open(fileDirPathInputName, "rb") as f:
        dict3C = pickle.load(f)
        
    recIm3C = dict3C["recIm3C"]

    maxMovRec = dict3C['maxMovementRec'];
    labelFl = idxLabel[:, fl*50 : fl*50+50]
    
    
    for i in range(0, recIm3C.shape[0]-1, skipFrameNumb):    
                
        im3CRaw = recIm3C[i,:,:,:]/1
        
        if np.count_nonzero(im3CRaw[:,:,0])>6400:            
            im3CRaw[:,:,0] = np.zeros((80,80))
        
        if np.count_nonzero(im3CRaw[:,:,1])>800:            
            im3CRaw[:,:,1] = np.zeros((80,80))
        
        rgbArray = np.zeros((80,80,3), 'uint8')
        rgbArray[..., 0] = im3CRaw[:,:,0]
        rgbArray[..., 1] = im3CRaw[:,:,1]
        rgbArray[..., 2] = im3CRaw[:,:,2]
        im3C = Image.fromarray(rgbArray)
         

        if fl == 0 and i == 0:
    
            rtImRec[indIm,:,:,:] = im3C
            yi = np.zeros((1,10));
            yi[0,int(labelFl[0,i])]=1
            yRec = yi
            yiVect = labelFl[0,i]
            yVectRec = yiVect
            
            indIm=indIm+1
            
        if (fl > 0 or i > 0) and (maxMovRec[i] > thresholdMovement) and labelFl[0,i] != 7:
            
            imRandRotated = misc.imrotate(im3C,np.random.randint(360))
            
            rtImRec[indIm,:,:,:] = imRandRotated
            
            yi = np.zeros((1,10));
            yi[0,int(labelFl[0,i])]=1            
            yRec = np.vstack((yRec,yi))
            yiVect = labelFl[0,i]
            yVectRec = np.vstack((yVectRec,yiVect))
            
            indIm=indIm+1
            
            
        if maxMovRec[i] < thresholdMovement:
            print(maxMovRec[i]);print('No movement detected')

            
Xin = rtImRec[0:indIm,:,:,:]
       

#Train with ConvNet

y=yVectRec
y=y[:,0]

Xin = Xin/256 #normalize images to 0-1

fShf, lShf = shuffle(Xin, y, random_state=0)
XShf = fShf
yShf = np.transpose(lShf)

XTrain = XShf[0:int(np.shape(Xin)[0]/3),:,:,:] #use 1/3 of the images for training
yTrain = yShf[0:int(np.shape(Xin)[0]/3)]


##########################################################################
#Convolutional Network Structure:

model = Sequential()

model.add(Conv2D(16, (5, 5), input_shape=Xin.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # 

model.add(Dense(128))
model.add(Activation('relu'))

model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(XTrain, yTrain, batch_size=32, epochs=20, validation_split=0.1)

model.save('modelConv2ABRS_3C') #save the graph and weights of the trained CNN to be used for classification


##########################################################################

predictionsProb = model.predict(Xin)

predictionLabel = np.zeros((1,np.shape(predictionsProb)[0]))
predictionLabel[0,:] = np.argmax(predictionsProb,axis=1) #this is the ethogram of the training data