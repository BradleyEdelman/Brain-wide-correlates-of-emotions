# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:05:10 2020

@author: bedelman
"""

# %%
# Single stimulus data

import os
import re
from os.path import isfile, join
from shutil import copyfile
import cv2


video = 0

# Specify data path details
Data = r"C:\Users\Bedelman\Videos"
Date = "20200923"
Mouse = "mouse_0026"
Stim = ['neutral', 'sucrose', 'salt', 'quinine', 'shock']
Stim = ['neutral', 'sucrose', 'quinine', 'salt', 'shock']
# Stim = ['neutral', 'sucrose']

path = Data + "\\" + Date + "\\" + Mouse
fold = os.listdir(path)
for i in range(0, len(Stim)):
    fold = [x for x in fold if Stim[i] not in x]

# create inividual stimulus folders for smaller data subsets
camera = ['BODY', 'FACE']
data_fold = []
for i in range(0, len(Stim)):
    data_fold.append(path + "\\" + Stim[i])
    for j in range(0, len(camera)):
        data_fold2 = data_fold[i] + "\\" + camera[j]
        if not os.path.exists(data_fold2):
            os.makedirs(data_fold2)

# stimulus start indices
# L = 60 # length of stimulus (frames) (seconds *30)
Pad = 2700 # length of baseline (frames) (seconds *30)
N = [900, 1800, 2700]
S1 = [5400, 9060, 12720]
S2 = [18180, 21840, 25500]
S3 = [30960, 34620, 38280]
S4 = [43740, 47400, 51060]
idx = [N, S1, S2, S3, S4]

idxtrial = []
for i in range(0, len(Stim)):
    
    if i == 0:
        idxtrial.append(list(range(0, idx[i][-1])))
    else:
        idxtrial.append(list(range(idx[i][0] - Pad, idx[i][-1] + Pad)))       
        


# each acquisition run
for i in range(0, len(fold)):
    
    stim_fold = path + "\\" + fold[i]
    
    # body and face cameras
    
    for j in range(0, len(camera)):
        
        cam_fold = stim_fold + "\\" + camera[j]
        #cam_rename_fold = stim_fold + "\\" + camera_rename[j]
        onlyfiles = [x for x in os.listdir(cam_fold) if isfile(join(cam_fold, x))  and ".jpg" in x]
        
        # rename files to simple numbering in new folder
        NUM = []
        for k in range(0, len(onlyfiles)):
            numbers = re.findall('[0-9]+', onlyfiles[k])
            NUM.append(int(numbers[-1]))
            
        onlyfiles_sorted = [x for _,x in sorted(zip(NUM,onlyfiles))]
        
        minidx = min(NUM)
        if minidx == 0:
            for k in range(0, len(Stim)):
                for l in range(0, len(idxtrial[k])):
                    src = cam_fold + "\\" + onlyfiles_sorted[idxtrial[k][l]]
                    dst = data_fold[k] + "\\" + camera[j] + "\\" + onlyfiles_sorted[idxtrial[k][l]]
                    
                    if not isfile(dst):
                            copyfile(src, dst)
        
                if video == 1:
                    Vid = data_fold[k] + "\\" + camera[j] + "\\" + camera[j] + ".avi"
                    if not os.path.isfile(Vid):
                    
                        img_array = []
                        for m in range(0, len(idxtrial[k])):
                           if ".jpg" in onlyfiles_sorted[idxtrial[k][m]]:
                               img = cv2.imread(data_fold[k] + "\\" + camera[j] + "\\" + onlyfiles_sorted[idxtrial[k][m]])
                               height, width, layers = img.shape
                               size = (width,height)
                               img_array.append(img)
                           
                           out = cv2.VideoWriter(Vid, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
                           
                        for m in range(0, len(img_array)):
                           out.write(img_array[m])
                        out.release()

        
        
        
        