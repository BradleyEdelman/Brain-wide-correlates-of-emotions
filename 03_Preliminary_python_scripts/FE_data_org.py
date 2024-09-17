# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 12:05:10 2020

@author: bedelman
"""

# %%
# Single stimulus data

import os
import re
from os.path import isfile, isdir, join
from shutil import copyfile

# Specify data path details
Data = r"C:\Users\Bedelman\Videos\Facial_Exp"
Date = "20201201"
Mouse = "mouse_0043"

path = Data + "\\" + Date + "\\" + Mouse
fold = [x for x in os.listdir(path) if isdir(join(path, x))]

# each acquisition run
for i in range(0, len(fold)):
    
    stim_fold = path + "\\" + fold[i]
    
    # body and face cameras
    camera = ['BODY', 'FACE']
    # camera_rename = ['body', 'face']
    for j in range(0, len(camera)):
        
        cam_fold = stim_fold + "\\" + camera[j]
        onlyfiles = [x for x in os.listdir(cam_fold) if isfile(join(cam_fold, x))  and ".jpg" in x]
        
        # rename files to simple numbering
        NUM = []
        for k in range(0, len(onlyfiles)):
        
            src = cam_fold + "\\" + onlyfiles[k]
            if isfile(src):
                
                numbers = re.findall('[0-9]+', onlyfiles[k])
                NUM.append(int(numbers[-1]))
        
        # if file count doesnt start at 0, rename again since stimulus indices assume starting at 0
        minidx = min(NUM)
        if minidx != 0:
            R = range(len(onlyfiles)-1, -1, -1)
            
            # rename files so first one starts at 0
            for k in R:
                
                src = cam_fold + "\\" + onlyfiles[k]
                if isfile(src):
                    
                    numbers = re.findall('[0-9]+', onlyfiles[k])
                    dst = cam_fold + "\\" + camera[j] + "-" + numbers[0] + "-" + str(int(numbers[1]) - minidx) + ".jpg"
                    os.rename(src,dst)
        
        if i == 0 and "neutral" not in fold:
            # if no neutral folder, use
            neutral_fold = path + "\\neutral\\" + camera[j]
            if not os.path.exists(neutral_fold):
                os.makedirs(neutral_fold)
            
            onlyfiles = [x for x in os.listdir(cam_fold) if isfile(join(cam_fold, x))  and ".jpg" in x]
             # rename files to simple numbering
            NUM = []
            for k in range(0, len(onlyfiles)):
            
                src = cam_fold + "\\" + onlyfiles[k]
                if isfile(src):
                    
                    numbers = re.findall('[0-9]+', onlyfiles[k])
                    NUM.append(int(numbers[-1]))
            
            onlyfiles_sorted = [x for _,x in sorted(zip(NUM,onlyfiles))]
            for k in range(1000, 3000):
                src2 = cam_fold + "\\" + onlyfiles_sorted[k]
                dst2 = neutral_fold + "\\" + onlyfiles_sorted[k]
                if isfile(src2) and not isfile(dst2):
                    copyfile(src2,dst2)

            
        
        
        