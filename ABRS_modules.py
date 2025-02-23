#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Primoz Ravbar UCSB
# Licensed under BSD 2-Clause [see LICENSE for details]
# Written by Primoz Ravbar

"""
Modified on Sat Oct  5 10:02:58 2019
@author: Augusto Escalante
"""
#This file contains functions used with ABRS


import numpy as np
import scipy
from scipy import misc #pip install pillow
import pickle
import cv2
import os
from scipy.signal import savgol_filter

def create_ST_image(cfrVectRec, roi):

    shapeCfrVectRec = cfrVectRec.shape;
    rw = np.zeros((1,shapeCfrVectRec[1])); 
    M1 = np.concatenate((cfrVectRec,rw), axis=0);
    M2 = np.concatenate((rw,cfrVectRec), axis=0);

    dM = M1 - M2;
    dM = dM[1:16,:];

    dM1 = np.concatenate((dM,rw), axis=0);
    dM2 = np.concatenate((rw,dM), axis=0);
    dM1M2 = dM1*dM2;
    MST = dM1M2*0;
    MST[dM1M2<0] = dM1M2[dM1M2<0];
    

    sM = np.sum(np.absolute(MST), axis=0);
    
    I = np.reshape(sM,(roi,roi));
	 
    return I, sM, MST;

def center_of_gravity(cfrVectRec):

    sh = np.shape(cfrVectRec);

    F=np.absolute(np.fft.fft(cfrVectRec,axis=0))
    
    av = np.zeros((1,sh[0]));
    av[0,:] = np.arange(1,sh[0]+1);
    A = np.repeat(av,sh[1],axis=0);

    FA = F*np.transpose(A);

    sF = np.sum(F,axis=0);
    sFA = np.sum(FA,axis=0);

    cG = sFA/sF;


    return cG

def center_of_gravity2(cfrVectRec):

    F=np.absolute(np.fft.fft(cfrVectRec,axis=0))

    sumF = np.sum(F,axis=0);
    Fnorm = F/sumF;
    F=Fnorm;
    
    shF = np.shape(F);
    halfFreq = int(np.ceil(shF[0]/2));

    F1 = F[0:halfFreq,:];
    
    shF1 = np.shape(F1);
    
    av = np.zeros((1,shF1[0]));
    av[0,:] = np.arange(1,shF1[0]+1);
    A = np.repeat(av,shF1[1],axis=0);

    FA = F1*np.transpose(A);

    sF = np.sum(F1,axis=0);
    sFA = np.sum(FA,axis=0);

    cG = sFA/sF

    return cG, F1

def center_of_gravity3(cfrVectRec):

    shX = np.shape(cfrVectRec);

    zeroPdd = 9;

    Xz=np.vstack((np.zeros((zeroPdd,shX[1])),cfrVectRec));
    Xz2 = np.vstack((Xz,np.zeros((zeroPdd,shX[1]))));

    F=np.absolute(np.fft.fft(Xz2,axis=0));

    shF = np.shape(F);
    halfFreq = int(np.ceil(shF[0]/2));
    hiCutOff = 0;

    F1 = F[0:halfFreq-hiCutOff,:];
    shF1 = np.shape(F1);

    sumF1 = np.sum(F1,axis=0);
    Fnorm = F1/sumF1;
    F1=Fnorm;
    
    av = np.zeros((1,shF1[0]));
    av[0,:] = np.arange(1,shF1[0]+1);
    A = np.repeat(av,shF1[1],axis=0);

    FA = F1*np.transpose(A);

    sF = np.sum(F1,axis=0);
    sFA = np.sum(FA,axis=0);

    cG = sFA/sF   

    return cG, F1

def create_3C_image (cfrVectRec, CVNsize):
    
    cG=center_of_gravity(cfrVectRec);

    averageSubtFrameVecRec = subtract_average(cfrVectRec,0)

    imRaw = np.reshape(cfrVectRec[1,:],(CVNsize,CVNsize));
    imRaw = imRaw.astype('uint')

    imDiff = np.reshape(cfrVectRec[1,:]-cfrVectRec[0,:],(CVNsize,CVNsize));
    imDiffAbs = np.absolute(imDiff)
    maxImDiffAbs = np.max(np.max(imDiffAbs))
    imDiffCl = np.zeros((CVNsize,CVNsize))
    imDiffCl[imDiffAbs > maxImDiffAbs/10] = imDiff[imDiffAbs > maxImDiffAbs/10]
    imDiffClNorm = imDiffCl/maxImDiffAbs
                                
    totVar = np.sum(np.absolute(averageSubtFrameVecRec),axis=0)
    imVar = np.reshape(totVar,(CVNsize,CVNsize))
    imVarNorm = imVar/np.max(np.max(imVar))
    imVarBin = np.zeros((CVNsize,CVNsize))
    imVarBin[imVarNorm > 0.10] = 1;
                                
    I = np.reshape(cG,(CVNsize,CVNsize))*imVarBin;
    I = np.nan_to_num(I);
                                
    imSTsm = smooth_2d (I, 3)
    sM = imSTsm.reshape((CVNsize*CVNsize))
    sM = np.nan_to_num(sM)

    if np.max(sM) > 0:
        sMNorm = sM/np.max(sM)
    else:
        sMNorm = sM

    I_RS = np.reshape(sMNorm,(CVNsize,CVNsize))

    imDiffAbs = np.absolute(imDiffClNorm)

    imDiffClNeg = np.zeros((CVNsize,CVNsize))
    imDiffClPos = np.zeros((CVNsize,CVNsize))
    imDiffClNeg[imDiffClNorm<0] = np.absolute(imDiffClNorm[imDiffClNorm<0])
    imDiffClPos[imDiffClNorm>0] = imDiffClNorm[imDiffClNorm>0]

    imDiffClNormNeg = imDiffClNeg/np.max(np.max(imDiffClNeg))
    imDiffClNormPos = imDiffClPos/np.max(np.max(imDiffClPos))
    

    rgbArray = np.zeros((CVNsize,CVNsize,3), 'uint8')
    rgbArray[..., 0] = I_RS*255 #blue channel for cv2.imshow()/ red channel for plt.imshow(): Difference with average of windowST frames
    rgbArray[..., 1] = imDiffAbs*255 #green channel for cv2.imshow() and plt.imshow(): Difference with previous frame
    rgbArray[..., 2] = imRaw*255 #red channel for cv2.imshow()/ blue channel for plt.imshow()

    im3C = rgbArray

    return im3C 
    

def subtract_average(frameVectRec,dim):

    shFrameVectRec = np.shape(frameVectRec);
    averageSubtFrameVecRec = np.zeros((shFrameVectRec[0],shFrameVectRec[1]));

    if dim == 0:
      averageVect = np.mean(frameVectRec,0);

    if dim == 1:
      averageVect = np.mean(frameVectRec,1);
      
    if dim == 0:
        for i in range(0,shFrameVectRec[0]):
           averageSubtFrameVecRec[i,:] = frameVectRec[i,:] - averageVect;

    if dim == 1:
        for i in range(0,shFrameVectRec[1]):
           averageSubtFrameVecRec[:,i] = frameVectRec[:,i] - averageVect;       

    return averageSubtFrameVecRec   

        

def read_frames(startFrame, endFrame, file_name, newSize):
#Modified to work with non-square movie frames
    
    cap = cv2.VideoCapture(file_name)
        
    print(file_name)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if height == width:
        pass
    if height < width:
        pad = width-height
    if height > width:
        pad = height-width
    
    for i in range(startFrame, endFrame):

        cap.set(1,i);
        ret, frame = cap.read() #
       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert frame to gray
        
        # Pad frame to make it square adding black pixels (frame,top,bottom,left,right)
        if height == width:
            gray = gray;
        if height < width:
            gray = cv2.copyMakeBorder(gray,0,pad,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
        if height > width:
            gray = cv2.copyMakeBorder(gray,0,0,0,pad,cv2.BORDER_CONSTANT,value=[0,0,0])
        
        if newSize[0] != height or newSize[0] != width: #Resize frame to newSize
            rs = cv2.resize(gray,(newSize[0],newSize[1]));
        if newSize[0] == height or newSize[0] == width: #No resizing:
            rs = gray;
        #Reshape the frame into a 1 row, many columns (for newSize=400: 400*400=160000)
        frameVect = rs.reshape(1,newSize[0]*newSize[1]);
        #Make the vector type float
        frameVectFloat = frameVect.astype(float);    

        if i == startFrame:
            frRec = frameVectFloat;
        if i > startFrame:
            frRec = np.vstack((frRec,frameVectFloat));

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()    
        
    return frRec;


def read_frames2(startFrame, endFrame, file_name, newSize):

    
    frRec = np.zeros((endFrame,newSize[0]*newSize[1]))
    
    
    cap = cv2.VideoCapture(file_name)

    print(file_name)

    for i in range(startFrame, endFrame):

        cap.set(1,i);
        ret, frame = cap.read() #

       
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert frame to gray
        
        rs = cv2.resize(gray,(newSize[0],newSize[1]))
        frameVect = rs.reshape(1,newSize[0]*newSize[1])
        frameVectFloat = frameVect.astype(float)

        frRec[i,:] = frameVectFloat

    

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()    
        
    return frRec;


def getting_frame_record(frRec, startWin, endWin, fb, newSize, roi, CVNsize):

    #Subdivide (or not, fb==0) the video frame in order to analyze each arena (for the authors setup that consist on
    #4 plates each with a fly in the same frame)        
    for i in range(startWin,endWin):

            frame = frRec[i,:];
            #Nothing changes from frame to gray, they are identical columns
            gray = frame.reshape(newSize[0]*newSize[1]);
            
            if fb == 0:
                rf = gray;
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

    #Find the index of the first pixel in the frame that shows the highest difference in intensity respect to the previous frame
    indMaxDiff = np.argmax(frameDiffComm);
    
    rowMaxDiff = np.floor(indMaxDiff/int(newSize[0]));
    colMaxDiff = indMaxDiff - (rowMaxDiff*int(newSize[0]));

    rowMaxDiff = rowMaxDiff.astype(int); 
    colMaxDiff = colMaxDiff.astype(int);

    #Find the value of the pixel in the frame that shows the highest difference in intensity respect to the previous frame
    maxMovement = np.max(frameDiffComm);

    posDic = {"xPos" : colMaxDiff, "yPos" : rowMaxDiff};
    

    for i in range(0,(endWin-startWin)):
           
           #Make frameVectFloatRec square
           rs = frameVectFloatRec[i,:].reshape(int(newSize[0]),int(newSize[0]))
           #Calculate a roixroi square around the pixel of maximum intensity difference 
           bottomOvershot=0
           rightOvershot=0
           
           topEdge = rowMaxDiff-int(roi*0.5);
           if topEdge < 0:
               topEdge=0;
           bottomEdge = rowMaxDiff+int(roi*0.5);
           if bottomEdge > int(newSize[0]):
               bottomOvershot = bottomEdge-int(newSize[0])
               bottomEdge=int(newSize[0]);
           leftEdge = colMaxDiff-int(roi*0.5);
           if leftEdge < 0:
               leftEdge=0;
           rightEdge = colMaxDiff+int(roi*0.5);
           if rightEdge > int(newSize[0]):
               rightOvershot = rightEdge-int(newSize[0])
               rightEdge=int(newSize[0]);
           

           #Select the roixroi square from the frame
           cfr = rs[topEdge:bottomEdge,leftEdge:rightEdge];
           shapeCfr = cfr.shape;

           #Correct (adding zeros) to make a square shape in case it is not roixroi due to negative values in above section substractions
           if topEdge == 0:
               rw = np.zeros((np.absolute(shapeCfr[0]-roi),shapeCfr[1]))
               cfr = np.vstack((rw,cfr))
               shapeCfr = cfr.shape; 
           if bottomOvershot > 0:
               rw = np.zeros((np.absolute(shapeCfr[0]-roi),shapeCfr[1]))
               cfr = np.vstack((cfr,rw))
               shapeCfr = cfr.shape; 
           if leftEdge == 0:
               col = np.zeros((shapeCfr[0], np.absolute(shapeCfr[1]-roi)))
               cfr = np.hstack((col,cfr))
               shapeCfr = cfr.shape; 
           if rightOvershot > 0:
               col = np.zeros((shapeCfr[0], np.absolute(shapeCfr[1]-roi)))
               cfr = np.hstack((cfr,col))
               shapeCfr = cfr.shape;
           
           #Resize roixroi to CVNsizexCVNsize:
           smallcfr = cv2.resize(cfr,(CVNsize,CVNsize));
           cfrVect = smallcfr.reshape(1,CVNsize*CVNsize);
           cv2.destroyAllWindows();
           
           if i == 0:
               cfrVectRec = cfrVect;
           if i > 0:
               cfrVectRec = np.vstack((cfrVectRec,cfrVect));
            

    return posDic, maxMovement, cfrVectRec, frameVectFloatRec;   



def find_movement_in_fb(rawFrRec, startWin, endWin, fb, newSize, roi):

    sh = np.shape(rawFrRec);
    oldSize = [int(np.sqrt(sh[1])),int(np.sqrt(sh[1]))];

    ratioDev = oldSize[0]/newSize[0];
    print(ratioDev)

    for i in range(startWin,endWin):


            rawFrameVect = rawFrRec[i,:];            
            rawFrame = rawFrameVect.reshape(oldSize[0],oldSize[1]);
            resizedFrame = cv2.resize(rawFrame,(newSize[0],newSize[1]));
            

            if fb == 1:
                rf = resizedFrame[0:int(newSize[0]/2),0:int(newSize[0]/2)];
                
            if fb == 2:   
                rf = resizedFrame[0:200,200:400];
               
            if fb == 3:
                rf = resizedFrame[200:400,0:200];
               
            if fb == 4:    
                rf = resizedFrame[200:400,200:400];

            rs = rf
            
            frameVect = rs.reshape(1,int(newSize[0]/2)*int(newSize[1]/2));
            frameVectFloat = frameVect.astype(float);

            if i == startWin:
               previousFrame = frameVectFloat;
               frameDiffComm = previousFrame*0;
               frameVectFloatRec = frameVectFloat;
            
            if i > startWin:
                frameDiffComm = frameDiffComm + np.absolute(frameVectFloat - previousFrame);
                frameVectFloatRec = np.vstack((frameVectFloatRec,frameVectFloat));
                previousFrame = frameVectFloat;


    indMaxDiff = np.argmax(frameDiffComm);
    
    rowMaxDiff = np.floor(indMaxDiff/int(newSize[0]/2));
    colMaxDiff = indMaxDiff - (rowMaxDiff*int(newSize[0]/2));


    rowMaxDiff = int(rowMaxDiff);
    colMaxDiff = int(colMaxDiff);


    posDic = {"xPos" : int(colMaxDiff*ratioDev), "yPos" : int(rowMaxDiff*ratioDev)};

    
    for i in range(startWin,endWin):

            rawFrameVect = rawFrRec[i,:];            
            rawFrame = rawFrameVect.reshape(oldSize[0],oldSize[1]);

            if fb == 1:
                rawFBF = rawFrame[0:int(oldSize[0]/2),0:int(oldSize[0]/2)];
                
            if fb == 2:   
                rf = resizedFrame[0:200,200:400];
               
            if fb == 3:
                rf = resizedFrame[200:400,0:200];
               
            if fb == 4:    
                rf = resizedFrame[200:400,200:400];

            rowPos = posDic["yPos"]; colPos = posDic["xPos"];    

            topEdge = rowPos-(roi*0.5);
            if topEdge < 0:
                topEdge=0;
            bottomEdge = rowPos+(roi*0.5);
            if bottomEdge < 0:
               bottomEdge=0;
            leftEdge = colPos-(roi*0.5);
            if leftEdge < 0:
                leftEdge=0;
            rightEdge = colPos+(roi*0.5);
            if rightEdge < 0:
                rightEdge=0;

            zoomInFrame = rawFBF[topEdge:bottomEdge,leftEdge:rightEdge];
            
            shapeZoomInFrame = zoomInFrame.shape; 
            
            if shapeZoomInFrame[1] < roi:
                col = np.zeros((shapeZoomInFrame[0], np.absolute(shapeZoomInFrame[1]-roi)))
                zoomInFrame = np.concatenate((col,zoomInFrame), axis=1);
                shapeZoomInFrame = zoomInFrame.shape;
            if shapeZoomInFrame[0] < roi:
                rw = np.zeros((np.absolute(shapeZoomInFrame[0]-roi),shapeZoomInFrame[1]));
                zoomInFrame = np.concatenate((rw,zoomInFrame), axis=0);
                shapeZoomInFrame = zoomInFrame.shape;
    

            zoomInFrameVect = zoomInFrame.reshape(1,roi*roi);

            if i == 0:
                zoomInFrameVectRec = zoomInFrameVect;
            if i > 0:
                zoomInFrameVectRec = np.vstack((zoomInFrameVectRec,zoomInFrameVect));
                       

    return posDic, zoomInFrameVectRec



def discrete_radon_transform(image, steps):
    
    R = np.zeros((steps, len(image)), dtype='float64')
    for s in range(steps):
        rotation = misc.imrotate(image, -s*180/steps).astype('float64') #pip install pillow
        R[:,s] = sum(rotation)
    return R

def smooth_1d (M, winSm, axis = 1):

    Msm = scipy.signal.savgol_filter(M, winSm, polyorder=0, deriv=0, axis=axis, mode='interp')
    return Msm

def smooth_2d (M, winSm):
    
    Msm = savgol_filter(M, window_length=winSm, polyorder=0);
    return Msm

def etho2ethoAP (idx):

    sh = np.shape(idx);
    idxAP = np.zeros((1,sh[1]))

    for i in range(0,sh[1]):

       if idx[0,i] == 1 or idx[0,i] == 2:           
           idxAP[0,i] = 1;

       if idx[0,i] == 3 or idx[0,i] == 4 or idx[0,i] == 5:
           idxAP[0,i] = 2;

       if idx[0,i] == 6:           
           idxAP[0,i] = 3;    

    return idxAP

def create_LDA_training_dataset (dirPathFeatures,numbFiles):

    fileList = sorted(os.listdir(dirPathFeatures));

    for fl in range(0, numbFiles, 1):

        inputFileName = fileList[fl];
        print (inputFileName);

        featureMatDirPathFileName = dirPathFeatures + '\\' + inputFileName;

        with open(featureMatDirPathFileName, "rb") as f:
             STF_30_posXY_dict = pickle.load(f);

        featureMatCurrent = STF_30_posXY_dict["featureMat"];
        posMatCurrent = STF_30_posXY_dict["posMat"];
        maxMovementMatCurrent = STF_30_posXY_dict["maxMovementMat"];

        if fl == 0:
            
            featureMat =  featureMatCurrent;
            posMat = posMatCurrent;
            maxMovementMat = maxMovementMatCurrent;

        if fl > 0:

            featureMat = np.hstack((featureMat,featureMatCurrent));
            posMat = np.hstack((posMat, posMatCurrent));
            maxMovementMat = np.hstack((maxMovementMat, maxMovementMatCurrent));

    return featureMat, posMat, maxMovementMat


def balance_labels2d(y,X,limitCol):

    shY = np.shape(y)
    shX = np.shape(X)

    yBal = np.zeros((1,1))
    yNew = np.zeros((1,1))

    XIm = np.zeros((1,shX[1],shX[2],shX[3]))
    XBal = np.zeros((shX[0],shX[1],shX[2],shX[3]))

    '''
    limitCol = np.zeros((10,1))
    limitCol[0]=2;
    limitCol[1]=shY[0]/10;
    limitCol[2]=shY[0]/10;
    limitCol[3]=shY[0]/10;
    limitCol[4]=shY[0]/10;
    limitCol[5]=shY[0]*100;
    limitCol[6]=shY[0]/70;
    limitCol[7]=2;
    '''

    sumCol = np.zeros((10,1))

    ind=0;

    for i in range(0,shY[0]):

        behInd = int(y[i,0])
        sumCol[behInd]=sumCol[behInd]+1
        

        if sumCol[behInd] < limitCol[behInd]:
            
            ind=ind+1

            yNew[0,0] = behInd;
            XIm = X[i,:,:,:]

            yBal = np.vstack((yBal,yNew))
            XBal[ind,:,:,:] = XIm


    yBal = yBal[1:ind,:]        
    XBal = XBal[1:ind,:,:,:]     
            
    return  yBal,XBal  

def removeZeroLabelsFromTrainingData (label,data):


    shData = np.shape(data);
    shLabel = np.shape(label);

    newLabel = label*0;
    newData = data*0;

    ind = 0;

    for i in range(0,shData[1]):

        if label[0,i] != 0:    

            newLabel[0,ind] = label[0,i]; 
            newData[:,ind] = data[:,i];

            ind=ind+1;

    return newLabel,newData


def computeSpeedFromPosXY (posMat,halfWindow):

    sh = np.shape(posMat);

    speedMat = np.zeros((1,sh[1]));

    for i in range(halfWindow,sh[1]-halfWindow,1):

        x_sqr =(np.absolute(posMat[0,i+halfWindow]-posMat[0,i-halfWindow]))*(np.absolute(posMat[0,i+halfWindow]-posMat[0,i-halfWindow])); 
        y_sqr =(np.absolute(posMat[1,i+halfWindow]-posMat[1,i-halfWindow]))*(np.absolute(posMat[1,i+halfWindow]-posMat[1,i-halfWindow]));
        speedMat[0,i] = np.sqrt(x_sqr+y_sqr);

    return speedMat

                                                                                

    
