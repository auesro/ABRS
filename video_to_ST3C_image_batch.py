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
import pickle
import os
import platform
import tensorflow as tf
from tensorflow import keras
from ABRS_modules import getting_frame_record
from ABRS_modules import read_frames
from ABRS_modules import create_3C_image

OSplatform = platform.system()

###############################################################
#Parameters set by the user
###############################################################

frameRate = 30;

#Number of frames in one clip
clipStart = 0;
clipEnd = 199 
bufferSize = 50;
clipsNumberMax = 1; #

#Size to which resize the original video:
newSize = [400,400];

#Desired roi size around subject of interest (must be pair) = subarea of the original frame:
roi = 100;

#Desired final image size for training the Convolutional Neural Network:
CVNsize = 80; 

#Number of frames to calculate the higher scale spatiotemporal feature (red channel):
windowST = 16;

#fbList = [1,2,3,4]; # works for raw movies with 2x2 arenas (split the frames into 4)
#fbList = [1]; # one arena in the frame #AER: it will still subdivide the arena and take just the upper left square because of function getting_frame_record  
fbList = [0];

clipFirst = 0;
firstFolder = 0;

# select a ConvNet model to recognize the behavior from the ST-images; behavior labels will be stored; select 'none' for no model
modelName = 'none' # use 'none' or the name in the model (must be in the ABRS folder)
#modelName = 'modelConv2ABRS_3C_train_with_labelCS1fb1_SS_plus_LiManualLabel'


if OSplatform == 'Linux':
    
    rawVidDirPath = '/home/augustoer/ABRS/Test';

    dirPathOutput = '/home/augustoer/ABRS'; #ST-images and other data will be stored here

if OSplatform == 'Windows':

    rawVidDirPath = 'INSERT ROW VIDEO FOLDER PATH';

    dirPathOutput = 'INSERT THE OUTPUT FOLDER PATH'; #ST-images and other data will be stored here

if OSplatform == 'Darwin':
    
    rawVidDirPath = 'INSERT ROW VIDEO FOLDER PATH';

    dirPathOutput = 'INSERT THE OUTPUT FOLDER PATH'; #ST-images and other data will be stored here   
##################################################################
    


def video_clips_to_3C_image_fun (dirPathInput,dirPathOutput,fbList,clipStart,clipEnd,clipsNumber,bufferSize,windowST,modelName,OSplatform,roi,CVNsize):
    
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
                        
                    frRec = read_frames(startFrame, endFrame, fileDirPathInputName, newSize) #resizes frame to newSize

                    if clIndex == clips[0] and startFrame == clipStart:
                        frRecRemain = np.zeros((windowST,(int(newSize[0]*newSize[1])))); #160000=400*400
                        frRec = np.concatenate((frRecRemain,frRec), axis=0);

                    if clIndex > clips[0] or startFrame > clipStart:
                        frRec = np.concatenate((frRecRemain,frRec), axis=0);  

                    for fb in fbList:

                        recIm3C = np.zeros((bufferSize,CVNsize,CVNsize,3))
                        # recIm3C = np.zeros((bufferSize,80,80,3))

                        for w in range(0,frRec.shape[0]-windowST):

                            startWin = w;
                            endWin = startWin + windowST;
                       
                            posDic, maxMovement, cfrVectRec, frameVectFloatRec = getting_frame_record(frRec, startWin, endWin,fb, newSize, roi, CVNsize)

                            im3C = create_3C_image (cfrVectRec,CVNsize)
                            
                            if modelName != 'none':
                                predictBehavior = 1  # predict behavior from ST-image and store the label
                            else:
                                predictBehavior = 0
                            
                            if predictBehavior == 1:
                                                                
                                X_rs = np.zeros((1,CVNsize,CVNsize,3))
                                # X_rs = np.zeros((1,80,80,3))
        
                                X_rs[0,:,:,:]=im3C

                                X = X_rs/256  # normalize

                                predictionsProb = model.predict(X)
                                predictionLabel = np.zeros((1,np.shape(predictionsProb)[0]))
                                predictionLabel[0,:] = np.argmax(predictionsProb,axis=1)

                            #cv2.imshow('im3C',im3C)

                            recIm3C[w,:,:,:]=im3C

                            xPos = posDic["xPos"];
                            yPos = posDic["yPos"];
                                                       

                            if w == 0:
                                global xPosRec
                                xPosRec = xPos;
                                global yPosRec
                                yPosRec = yPos;
                                global maxMovementRec
                                maxMovementRec = maxMovement
                                    
                                if modelName != 'none': 
                                       global behPredictionRec 
                                       behPredictionRec = predictionLabel 
                                else: behPredictionRec = 0        
                                
                            if w > 0:
                                
                                xPosRec = np.vstack((xPosRec,xPos));
                                yPosRec = np.vstack((yPosRec,yPos));
                                maxMovementRec = np.vstack((maxMovementRec,maxMovement));
                                
                                if modelName != 'none': 
                                    behPredictionRec = np.vstack((behPredictionRec,predictionLabel))
                                else: behPredictionRec = 0     
                                   
                        xPosRec=0
                        yPosRec=0
                        dictPosRec = {"xPosRec" : xPosRec, "yPosRec" : yPosRec};
                        maxMovementRec=0
                        behPredictionRec=0
                        dictST = {"recIm3C" : recIm3C, "dictPosRec" : dictPosRec, "maxMovementRec" : maxMovementRec, "behPredictionRec" : behPredictionRec};
                        
                        nameSMRec = 'dict3C_' + fileName[0:-4] + '_Arena1_' + str('%06.0f' % bf) 
                       
                        if OSplatform == 'Linux':           
                            #newPath = dirPathOutput + '/' + fileName[0:-4] + '_fb' + str(fb)  #!!!!!!works with non-numbered clips (one clip per video recording)
                            newPath = dirPathOutput + '/' + fileName[0:-4] + '_Arena1' #!!!!!!works with numbered clips (multiple clips of the same video; clip names end with index number)
                        
                        if OSplatform == 'Windows':           
                            #newPath = dirPathOutput + '\\' + fileName[0:-4] + '_fb' + str(fb)  #!!!!!!works with non-numbered clips (one clip per video recording)
                            newPath = dirPathOutput + '\\' + fileName[0:-4] + '_Arena1' #!!!!!!works with numbered clips (multiple clips of the same video; clip names end with index number)

                        if OSplatform == 'Darwin':           
                            #newPath = dirPathOutput + '/' + fileName[0:-4] + '_fb' + str(fb)  #!!!!!!works with non-numbered clips (one clip per video recording)
                            newPath = dirPathOutput + '/' + fileName[0:-4] + '_Arena1' #!!!!!!works with numbered clips (multiple clips of the same video; clip names end with index number)
 

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

           
#Batch part:    
# videoFolderList = sorted(next(os.walk(rawVidDirPath))[2]); #list only the files, not directories
videoFolderList = sorted(os.listdir(rawVidDirPath)); #list of files and directories

sz = np.shape(videoFolderList); sizeVideoFolder = sz[0];

for fld in range(firstFolder, sizeVideoFolder):
    
    print(fld)

    currentVideoFolder = videoFolderList[fld];

    if currentVideoFolder[-5:] != 'Store' :

        dirPathInput = rawVidDirPath #+ '/' + currentVideoFolder;
        # clipList = sorted(next(os.walk(dirPathInput))[2]);
        clipList = sorted(os.listdir(dirPathInput));
        szClipList = np.shape(clipList);
        clipsNumber = szClipList[0];

        if clipsNumber > clipsNumberMax:
            clipsNumber = clipsNumberMax;

        video_clips_to_3C_image_fun (dirPathInput,dirPathOutput,fbList,clipStart,clipEnd,clipsNumber,bufferSize,windowST,modelName,OSplatform,roi,CVNsize);
        
        
# #Batch part:    
# videoFileList = sorted(next(os.walk(rawVidDirPath))[2]); #list only the files, not directories


# for nf in range(1,len(videoFileList)):
    
#     currentFile = videoFileList[nf];
#     dirPathInput = rawVidDirPath
#     clipsNumber = len(videoFileList);
    
#     video_clips_to_3C_image_fun (dirPathInput,dirPathOutput,fbList,clipStart,clipEnd,clipsNumber,bufferSize,windowST,modelName,OSplatform,roi,CVNsize);
