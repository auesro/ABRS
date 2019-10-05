#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from scipy import ndimage
from scipy import misc
import pickle
import time
import matplotlib.pyplot as plt
import cv2
import os
import platform
import tensorflow as tf
from tensorflow import keras

from ABRS_modules import discrete_radon_transform
from ABRS_modules import create_ST_image
from ABRS_modules import smooth_2d
from ABRS_modules import center_of_gravity
#from ABRS_modules import subplot_images
from ABRS_modules import getting_frame_record
from ABRS_modules import read_frames
from ABRS_modules import read_frames2
from ABRS_modules import subtract_average
from ABRS_modules import create_3C_image



def video_clips_to_3C_image_fun (dirPathInput,dirPathOutput,fbList,clipStart,clipEnd,clipsNumber,bufferSize,windowST,modelName,OSplatform):

    OSplatform = platform.system()
    
    if modelName != 'none':
    
        model = keras.models.load_model(modelName)

    clips = np.arange(clipFirst,clipsNumber,1);

    fileList = os.listdir(dirPathInput);

    clIndex = 0;

    for cl in clips:


        fileName = fileList[cl];

        ext = fileName[-3:];

        if (ext == 'avi' or ext == 'mov' or ext == 'mp4') == True:
            if OSplatform == 'Linux':           
                fileDirPathInputName = dirPathInput + '/' + fileName;
            if OSplatform == 'Windows':           
                fileDirPathInputName = dirPathInput + '\\' + fileName;
            if OSplatform == 'Darwin':           
                fileDirPathInputName = dirPathInput + '/' + fileName;
                
            r = np.arange(0,clipEnd,bufferSize)

            for bf in r:

                    startFrame = bf;
                    endFrame = startFrame + bufferSize;
                        
                    frRec = read_frames(startFrame, endFrame, fileDirPathInputName, [400,400])

                    if clIndex == clips[0] and startFrame == clipStart:
                        frRecRemain = np.zeros((windowST,160000));
                        frRec = np.concatenate((frRecRemain,frRec), axis=0);

                    if clIndex > clips[0] or startFrame > clipStart:
                        frRec = np.concatenate((frRecRemain,frRec), axis=0);  

                    for fb in fbList:

                        recIm3C = np.zeros((50,80,80,3))

                        for w in range(0,frRec.shape[0]-windowST):

                            startWin = w;
                            endWin = startWin + windowST;
                       
                            posDic, maxMovement, cfrVectRec, frameVectFloatRec = getting_frame_record(frRec, startWin, endWin,fb)


                            im3C = create_3C_image (cfrVectRec)
                            #print(im3C[30,30,2])
                            
                            if modelName != 'none':
                                predictBehavior = 1  # predict behavior from ST-image and store the label
                            else:
                                predictBehavior = 0
                            
                            if predictBehavior == 1:
                                                                
                                X_rs = np.zeros((1,80,80,3))
        
                                X_rs[0,:,:,:]=im3C

                                X = X_rs/256  # normalize

                                predictionsProb = model.predict(X)
                                predictionLabel = np.zeros((1,np.shape(predictionsProb)[0]))
                                predictionLabel[0,:] = np.argmax(predictionsProb,axis=1)
                    
                    

                            cv2.imshow('im3C',im3C)

                            recIm3C[w,:,:,:]=im3C

                            xPos = posDic["xPos"];
                            yPos = posDic["yPos"];
                                                       

                            if w == 0:
                            
                                xPosRec = xPos;
                                yPosRec = yPos;
                                maxMovementRec = maxMovement
                                    
                                if modelName != 'none': 
                                       behPredictionRec = predictionLabel 
                                else: behPredictionRec = 0        
                                
                            if w > 0:
                                
                                xPosRec = np.vstack((xPosRec,xPos));
                                yPosRec = np.vstack((yPosRec,yPos));
                                maxMovementRec = np.vstack((maxMovementRec,maxMovement));
                                
                                if modelName != 'none': 
                                    behPredictionRec = np.vstack((behPredictionRec,predictionLabel))
                                else: behPredictionRec = 0     
                                   
                        
                       
                        dictPosRec = {"xPosRec" : xPosRec, "yPosRec" : yPosRec};

                        dictST = {"recIm3C" : recIm3C, "dictPosRec" : dictPosRec, "maxMovementRec" : maxMovementRec, "behPredictionRec" : behPredictionRec};
                        
                        nameSMRec = 'dict3C_' + fileName[0:-4] + '_fb' + str(fb) + '_bf_' + str('%06.0f' % bf) 
                       
                        if OSplatform == 'Linux':           
                            #newPath = dirPathOutput + '/' + fileName[0:-4] + '_fb' + str(fb)  #!!!!!!works with non-numbered clips (one clip per video recording)
                            newPath = dirPathOutput + '/' + fileName[0:-6] + '_fb' + str(fb) #!!!!!!works with numbered clips (multiple clips of the same video; clip names end with index number)
                        
                        if OSplatform == 'Windows':           
                            #newPath = dirPathOutput + '\\' + fileName[0:-4] + '_fb' + str(fb)  #!!!!!!works with non-numbered clips (one clip per video recording)
                            newPath = dirPathOutput + '\\' + fileName[0:-6] + '_fb' + str(fb) #!!!!!!works with numbered clips (multiple clips of the same video; clip names end with index number)

                        if OSplatform == 'Darwin':           
                            #newPath = dirPathOutput + '/' + fileName[0:-4] + '_fb' + str(fb)  #!!!!!!works with non-numbered clips (one clip per video recording)
                            newPath = dirPathOutput + '/' + fileName[0:-6] + '_fb' + str(fb) #!!!!!!works with numbered clips (multiple clips of the same video; clip names end with index number)
 

                        if not os.path.exists(newPath):
                            os.mkdir(newPath);
                        if OSplatform == 'Linux': 
                            fileDirPathOutputName = newPath + '/' + nameSMRec;
                        if OSplatform == 'Windows': 
                            fileDirPathOutputName = newPath + '\\' + nameSMRec;
                        if OSplatform == 'Darwin': 
                            fileDirPathOutputName = newPath + '/' + nameSMRec;    
                                   
                        with open(fileDirPathOutputName, "wb") as f:
                            pickle.dump(dictST, f)

                             
                    frRecSh = frRec.shape;
                    frRecRemain = frRec[bufferSize:frRecSh[0],:]
            clIndex = clIndex +1;
            print(clIndex)


OSplatform = platform.system()

frameRate = 30;

clipStart = 0
clipEnd = 999 # number of frames in one clip


clipsNumberMax = 51; #

#fbList = [1,2,3,4]; # works for raw movies with 2x2 arenas (split the frames into 4)
fbList = [1]; # one arena in the frame

clipFirst = 0;

bufferSize = 50;

# select a ConvNet model to recognize the behavior from the ST-images; behavior labels will be stored; select 'none' for no model

modelName = 'none' # use 'none' or the name in the model (must be in the ABRS folder)

#modelName = 'modelConv2ABRS_3C_train_with_labelCS1fb1_SS_plus_LiManualLabel'

firstFolder = 0; # 

if OSplatform == 'Linux':
    
    rawVidDirPath = '/home/auesro/Desktop/ABRS Test';windowST = 16;

    dirPathOutput = '/home/auesro/Desktop/Store'; #ST-images and other data will be stored here

if OSplatform == 'Windows':

    rawVidDirPath = 'INSERT ROW VIDEO FOLDER PATH';windowST = 16;

    dirPathOutput = 'INSERT THE OUTPUT FOLDER PATH'; #ST-images and other data will be stored here

if OSplatform == 'Darwin':
    
    rawVidDirPath = 'INSERT ROW VIDEO FOLDER PATH';windowST = 16;

    dirPathOutput = 'INSERT THE OUTPUT FOLDER PATH'; #ST-images and other data will be stored here   

videoFolderList = sorted(os.listdir(rawVidDirPath));

sz = np.shape(videoFolderList);sizeVideoFolder = sz[0];

for fld in range(firstFolder, sizeVideoFolder):
    
    print(fld)

    currentVideoFolder = videoFolderList[fld];

    if currentVideoFolder[-5:] != 'Store' :

        dirPathInput = rawVidDirPath #+ '\\' + currentVideoFolder;
        clipList = sorted(os.listdir(dirPathInput));szClipList = np.shape(clipList);
        clipsNumber = szClipList[0];

        if clipsNumber > clipsNumberMax:
            clipsNumber = clipsNumberMax;

        video_clips_to_3C_image_fun (dirPathInput,dirPathOutput,fbList,clipStart,clipEnd,clipsNumber,bufferSize,windowST,modelName,OSplatform);

