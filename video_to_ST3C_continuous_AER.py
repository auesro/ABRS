#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#ABRS_labelMaker

# Copyright (c) 2019 Primoz Ravbar UCSB
# Licensed under BSD 2-Clause [see LICENSE for details]
# Written by Primoz Ravbar

"""
Edited on Sat Dec 14 17:49:37 2019
@author: auesro
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import os

import tkinter
from tkinter import filedialog

from ABRS_modules import getting_frame_record
from ABRS_modules import create_3C_image

###############################################################
#Parameters set by the user
###############################################################

#Size to which resize the original video (if equal to the longest dimension, 
#no resizing will take place (no resizing will result in slower processing 
#and apparently there is no resolution advantage given the final resizing to 80)):
newSize = [400,400];

#Desired roi size around subject of interest (must be pair) = subarea of the original frame:
roi = 80;

#Desired final image size for training the Convolutional Neural Network:
CVNsize = 80; 


startFrame = 300  # set this to any frame in the movie clip
endFrame = 350

#Number of frames to calculate the higher scale spatiotemporal feature (red channel):
windowST = 15; #~=0.5 seconds at 30 fps
# windowST = 10; #~=0.33 seconds at 30 fps
# windowST = 20; #~=0.66 seconds at 30 fps
# windowST = 30; #=1 second at 30 fps


#fbList = [1,2,3,4]; # works for raw movies with 2x2 arenas (split the frames into 4)
#fbList = [1]; # one arena in the frame #AER: it will still subdivide the arena and take just the upper left square because of function getting_frame_record  
fbList = 0;
###############################################################


# show an "Open" dialog box and return the path to the selected file
root = tkinter.Tk()
root.wm_withdraw()
fileDirPathInputName = filedialog.askopenfilename()
root.destroy()
root.mainloop()


cap = cv2.VideoCapture(fileDirPathInputName);

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if height < width:
    pad = width-height
if height > width:
    pad = height-width

prevFrame = np.zeros((newSize[0],newSize[0]))
frRec = np.zeros((windowST+1,newSize[0]*newSize[1]))
im3Crec = np.zeros(((endFrame-startFrame),CVNsize,CVNsize,3))

for frameInd in range(startFrame,endFrame,1):

    cap.set(1,frameInd)
    
    ret, frame = cap.read()

    if np.size(np.shape(frame)) >= 2:            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale
    else:
        print('Corrupt frame with less than 2 dimensions!!')
        gray = np.zeros((width, height)) # Fill the corrupt frame with black
        
    # Pad frame to make it square adding black pixels (frame,top,bottom,left,right)
    if height == width:
        gray2 = gray;
    if height < width:
        gray2 = cv2.copyMakeBorder(gray,0,pad,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
    if height > width:
        gray2 = cv2.copyMakeBorder(gray,0,0,0,pad,cv2.BORDER_CONSTANT,value=[0,0,0])
    
    #Resize frame to newSize if any of the dimensions is different from newSize    
    if newSize[0] != height or newSize[0] != width: 
        rs = cv2.resize(gray2,(newSize[0],newSize[1]));
    #If one of the dimensions is equal to newSize, no resizing is applied:
    if newSize[0] == height or newSize[0] == width:
        rs = gray2;

    currentFrame = rs.astype(float)/1;
    diffFrame = currentFrame - prevFrame;
    prevFrame = currentFrame;
    diffFrameAbs = np.absolute(diffFrame)

    frameVect = currentFrame.reshape(1,newSize[0]*newSize[1]);
    frameVectFloat = frameVect.astype(float);

    frRecShort = np.delete(frRec, 0, 0);
    frRec = np.vstack((frRecShort,frameVectFloat));
    
    posDic, maxMovement, cfrVectRec, frameVectFloatRec = getting_frame_record(frRec, 0, windowST, fbList, newSize, roi, CVNsize);
    im3CRaw = create_3C_image (cfrVectRec, CVNsize)
        
    rgbArray = np.zeros((CVNsize,CVNsize,3), 'uint8')
    rgbArray[..., 0] = im3CRaw[:,:,0]
    rgbArray[..., 1] = im3CRaw[:,:,1]
    rgbArray[..., 2] = im3CRaw[:,:,2]
    im3C = rgbArray

    indImage = frameInd-startFrame #Start saving first frame to position 0 independently of value of frameInd
    im3Crec[indImage,:,:,:]=im3C

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

newPath = fileDirPathInputName[0:-11] + '/' + 'Result'
if not os.path.exists(newPath):
    os.mkdir(newPath);

OutputFilePath = newPath + '/' + str('%06.0f' % startFrame) + '_' + str('%06.0f' % frameInd)
with open(OutputFilePath, "wb") as f:
    pickle.dump(im3Crec,f)