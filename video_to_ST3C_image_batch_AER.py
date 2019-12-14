#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 10:02:58 2019

@author: auesro
"""
import numpy as np
import pickle
import os
import platform
from tensorflow import keras
#from ABRS_modules import getting_frame_record
from ABRS_modules import read_frames
from ABRS_modules import create_3C_image
#from ABRS_modules import resize_to_80
import cv2

OSplatform = platform.system()

frameRate = 30;

clipStart = 0;
clipEnd = 99 # number of frames in one clip

newSize = [400,400];
clipsNumberMax = 2; #

#fbList = [1,2,3,4]; # works for raw movies with 2x2 arenas (split the frames into 4)
#fbList = [1]; # one arena in the frame
fbList = [0];

windowST = 5;

clipFirst = 0;

bufferSize = 50;

# select a ConvNet model to recognize the behavior from the ST-images; behavior labels will be stored; select 'none' for no model

modelName = 'none' # use 'none' or the name in the model (must be in the ABRS folder)

#modelName = 'modelConv2ABRS_3C_train_with_labelCS1fb1_SS_plus_LiManualLabel'

firstFolder = 0;

if OSplatform == 'Linux':
    
    rawVidDirPath = '/home/auesro/Desktop/ABRS Test';

    dirPathOutput = '/home/auesro/Desktop/Store'; #ST-images and other data will be stored here

if OSplatform == 'Windows':

    rawVidDirPath = 'INSERT ROW VIDEO FOLDER PATH';

    dirPathOutput = 'INSERT THE OUTPUT FOLDER PATH'; #ST-images and other data will be stored here

if OSplatform == 'Darwin':
    
    rawVidDirPath = 'INSERT ROW VIDEO FOLDER PATH';

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
#def video_clips_to_3C_image_fun (dirPathInput,dirPathOutput,fbList,clipStart,clipEnd,clipsNumber,bufferSize,windowST,modelName,OSplatform):
    
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
                    
                frRec = read_frames(startFrame, endFrame, fileDirPathInputName, newSize) #resizes frame to 400x400

                if clIndex == clips[0] and startFrame == clipStart:
                    frRecRemain = np.zeros((windowST,(int(newSize[0]*newSize[1])))); #160000=400*400
                    frRec = np.concatenate((frRecRemain,frRec), axis=0);

                if clIndex > clips[0] or startFrame > clipStart:
                    frRec = np.concatenate((frRecRemain,frRec), axis=0);  

                for fb in fbList:

                    recIm3C = np.zeros((bufferSize,80,80,3))

                    for w in range(0,frRec.shape[0]-windowST):

                        startWin = w;
                        endWin = startWin + windowST;
                   
                        #posDic, maxMovement, cfrVectRec, frameVectFloatRec = getting_frame_record(frRec, startWin, endWin,fb, newSize)
                        
                        for i in range(startWin,endWin):

                            frame = frRec[i,:];
                            gray = frame.reshape(newSize[0]*newSize[1]);
                            
                            if fb == 0:
                                rf = gray #np.resize(gray, (200,200));
                            if fb == 1:
                                rf = gray[0:200,0:200];
                                
                            if fb == 2:   
                                rf = gray[0:200,200:400];
                               
                            if fb == 3:
                                rf = gray[200:400,0:200];
                               
                            if fb == 4:    
                                rf = gray[200:400,200:400];
                
                            rs = rf   
                            
                            frameVect = rs.reshape(1,int(newSize[0])*int(newSize[1]));
                            frameVectFloat = frameVect.astype(float)
                
                 
                            if i == startWin:
                               previousFrame = frameVectFloat;
                               frameDiffComm = previousFrame*0;
                               frameVectFloatRec = frameVectFloat;
                            
                            if i > startWin:
                                frameDiffComm = frameDiffComm + np.absolute(frameVectFloat - previousFrame);
                                frameVectFloatRec = np.vstack((frameVectFloatRec,frameVectFloat));
                                previousFrame = frameVectFloat;
                
                
                        indMaxDiff = np.argmax(frameDiffComm);
                        
                        rowMaxDiff = np.floor(indMaxDiff/int(newSize[0]));
                        colMaxDiff = indMaxDiff - (rowMaxDiff*int(newSize[0]));
                    
                        rowMaxDiff = rowMaxDiff.astype(int);
                        colMaxDiff = colMaxDiff.astype(int);
                    
                        maxMovement = np.max(frameDiffComm);
                    
                        posDic = {"xPos" : colMaxDiff, "yPos" : rowMaxDiff};
                    
                
                        for i in range(0,(endWin-startWin)):
                            
                               rs = frameVectFloatRec[i,:].reshape(int(newSize[0]),int(newSize[1]))
                    
                               bottomOvershot=0
                               rightOvershot=0
                               
#                               topEdge = rowMaxDiff-40;
#                               if topEdge < 0:
#                                   topEdge=0;
#                               bottomEdge = rowMaxDiff+40;
#                               if bottomEdge > newSize[0]:
#                                   bottomOvershot = bottomEdge-int(newSize[0])
#                                   bottomEdge=int(newSize[0]);
#                               leftEdge = colMaxDiff-40;
#                               if leftEdge < 0:
#                                   leftEdge=0;
#                               rightEdge = colMaxDiff+40;
#                               if rightEdge > int(newSize[0]):
#                                   rightOvershot = rightEdge-int(newSize[0])
#                                   rightEdge=int(newSize[0]);
                    
                               topEdge = rowMaxDiff-50;
                               if topEdge < 0:
                                   topEdge=0;
                               bottomEdge = rowMaxDiff+50;
                               if bottomEdge > int(newSize[0]):
                                   bottomOvershot = bottomEdge-int(newSize[0])
                                   bottomEdge=int(newSize[0]);
                               leftEdge = colMaxDiff-50;
                               if leftEdge < 0:
                                   leftEdge=0;
                               rightEdge = colMaxDiff+50;
                               if rightEdge > int(newSize[0]):
                                   rightOvershot = rightEdge-int(newSize[0])
                                   rightEdge=int(newSize[0]);

                               cfr = rs[topEdge:bottomEdge,leftEdge:rightEdge];
                               shapeCfr = cfr.shape; 
                    
                    
#                               if topEdge == 0:
#                                   rw = np.zeros((np.absolute(shapeCfr[0]-80),shapeCfr[1]))
#                                   cfr = np.vstack((rw,cfr))
#                                   shapeCfr = cfr.shape; 
#                               if bottomOvershot > 0:
#                                   rw = np.zeros((np.absolute(shapeCfr[0]-80),shapeCfr[1]))
#                                   cfr = np.vstack((cfr,rw))
#                                   shapeCfr = cfr.shape; 
#                               if leftEdge == 0:
#                                   col = np.zeros((shapeCfr[0], np.absolute(shapeCfr[1]-80)))
#                                   cfr = np.hstack((col,cfr))
#                                   shapeCfr = cfr.shape; 
#                               if rightOvershot > 0:
#                                   col = np.zeros((shapeCfr[0], np.absolute(shapeCfr[1]-80)))
#                                   cfr = np.hstack((cfr,col))
#                                   shapeCfr = cfr.shape;
                    
#                               cfrVect = cfr.reshape(1,80*80);
                            
                               if topEdge == 0:
                                   rw = np.zeros((np.absolute(shapeCfr[0]-100),shapeCfr[1]))
                                   cfr = np.vstack((rw,cfr))
                                   shapeCfr = cfr.shape; 
                               if bottomOvershot > 0:
                                   rw = np.zeros((np.absolute(shapeCfr[0]-100),shapeCfr[1]))
                                   cfr = np.vstack((cfr,rw))
                                   shapeCfr = cfr.shape; 
                               if leftEdge == 0:
                                   col = np.zeros((shapeCfr[0], np.absolute(shapeCfr[1]-100)))
                                   cfr = np.hstack((col,cfr))
                                   shapeCfr = cfr.shape; 
                               if rightOvershot > 0:
                                   col = np.zeros((shapeCfr[0], np.absolute(shapeCfr[1]-100)))
                                   cfr = np.hstack((cfr,col))
                                   shapeCfr = cfr.shape;
                    
                               smallcfr = cv2.resize(cfr,(80,80))
                    
                               cfrVect = smallcfr.reshape(1,80*80);
                    
                               
                                
                               if i == 0:
                                   cfrVectRec = cfrVect;
                               if i > 0:
                                   cfrVectRec = np.vstack((cfrVectRec,cfrVect));

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
                
                

                        #cv2.imshow('im3C',im3C)

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
                    
                    nameSMRec = 'dict3C_' + fileName[0:-4] + '_Arena' + str(fb) + '_bf_' + str('%06.0f' % bf) 
                   
                    if OSplatform == 'Linux':           
                        #newPath = dirPathOutput + '/' + fileName[0:-4] + '_fb' + str(fb)  #!!!!!!works with non-numbered clips (one clip per video recording)
                        newPath = dirPathOutput + '/' + fileName[0:-4] + '_Arena' + str(fb) #!!!!!!works with numbered clips (multiple clips of the same video; clip names end with index number)
                    
                    if OSplatform == 'Windows':           
                        #newPath = dirPathOutput + '\\' + fileName[0:-4] + '_fb' + str(fb)  #!!!!!!works with non-numbered clips (one clip per video recording)
                        newPath = dirPathOutput + '\\' + fileName[0:-4] + '_Arena' + str(fb) #!!!!!!works with numbered clips (multiple clips of the same video; clip names end with index number)

                    if OSplatform == 'Darwin':           
                        #newPath = dirPathOutput + '/' + fileName[0:-4] + '_fb' + str(fb)  #!!!!!!works with non-numbered clips (one clip per video recording)
                        newPath = dirPathOutput + '/' + fileName[0:-4] + '_Arena' + str(fb) #!!!!!!works with numbered clips (multiple clips of the same video; clip names end with index number)
 

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




