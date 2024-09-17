# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:28:03 2020

@author: bedelman
"""

from distutils.dir_util import copy_tree
import os
import cv2
import re
import numpy as np
import glob
from os.path import isfile, isdir, join

<<<<<<< Updated upstream
path = r'\\nas6\datastore_brad$\Facial_Exp_fUS'
save_path = r'\\nas6\datastore_brad$\Facial_Exp_Movies\Fentanyl_test'
=======
path = r'D:\BRAD_FACIAL_EXP'
# save_path = r'C:\data\experiments\Facial_Exp_Movies\Miranda_FE_videos'
# path = r'\\nas6\datastore_brad$\Facial_Exp_FP'
save_path = r'\\nas6\datastore_brad$\Facial_Exp_Movies\Block_design_emotion'
>>>>>>> Stashed changes


camera = ['BODY', 'FACE', 'PUPIL']
camera_rename = ['body', 'face', 'pupil']

fold = os.listdir(path)
<<<<<<< Updated upstream
for  i in range(len(fold)-1,len(fold)):
=======
for  i in range(1,len(fold)):
>>>>>>> Stashed changes
    
    Day = path + "\\" + fold[i]
    if os.path.isdir(Day):
        day_fold = os.listdir(Day)
        for j in range(0, len(day_fold)):
            
            Mouse = Day + "\\" + day_fold[j]
            if os.path.isdir(Mouse):
                mouse_fold = os.listdir(Mouse)
                for k in range(0, len(mouse_fold)):
                               
                    Stim = Mouse + "\\" + mouse_fold[k]
                    if os.path.isdir(Stim):
                        stim_fold = os.listdir(Stim)
                        for l in range(0, len(stim_fold)):
                            
                            Cam = Stim + "\\" + stim_fold[l]
                            if os.path.isdir(Cam):
                                
                                onlyfiles = [x for x in os.listdir(Cam) if isfile(join(Cam, x)) and ".jpg" in x]
                                if onlyfiles:
                                
                                    NUM = []
                                    for m in range(0, len(onlyfiles)):
                                        numbers = re.findall('[0-9]+', onlyfiles[m])
                                        NUM.append(int(numbers[-1]))
                                        
                                    onlyfiles_sorted = [x for _,x in sorted(zip(NUM,onlyfiles))]
                                    del onlyfiles
                                    
                                    max_L = 20 # in minutes
                                    num_vid = int(np.ceil(len(onlyfiles_sorted)/(max_L*60*20)))
                                    for n in range(0, num_vid):
                                        # Vid = Cam + "\\" + camera[l] + '_' + str(n+1) + ".avi"
                                        Vid = save_path + "\\" + camera[l] + "\\" + day_fold[j] + '_' + fold[i] + '_' + mouse_fold[k] + '_' + camera[l] + '_' + str(n+1) + ".avi"
                                        print(Vid)

                                        if not os.path.isfile(Vid):
                                        
                                            img_array = []
                                            START = 0 +(n)*max_L*60*20
                                            END = max_L*60*20 + (n)*max_L*60*20
                                            if END > len(onlyfiles_sorted):
                                                END = len(onlyfiles_sorted)
                                            
                                            for m in range(START, END):
                                               # img = cv2.imread(Cam + "\\" + camera_rename[l] + "_" + str(NUM[m]) + ".jpg")
                                               if ".jpg" in onlyfiles_sorted[m]:
                                                   img = cv2.imread(Cam + "\\" + onlyfiles_sorted[m])
                                                   height, width, layers = img.shape
                                                   size = (width,height)
                                                   img_array.append(img)
                                               
                                               out = cv2.VideoWriter(Vid, cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
                                               
                                            for m in range(0, len(img_array)):
                                               out.write(img_array[m])
                                            out.release()
                                            
                                            del img_array 
                                 
                    