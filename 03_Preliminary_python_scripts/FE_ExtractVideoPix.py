# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:10:58 2020

@author: bedelman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import re
import os
from os.path import isfile, isdir, join
import FE_helper_func as FE
import scipy.io as sio


# %% 
# example image - full frame

# Specify data path details
multistim = 0 # 1 for yes, 0 for no
Data = r"C:\Users\Bedelman\Videos\Facial_Exp"
# Date = ["sample", "20200825", "20200826", "20200827", "20200923"]
# Mouse = [["mouse_0000", "mouse_0001"],["mouse_0008 - Copy", "mouse_0009"], ["mouse_0009", "mouse_0010", "mouse_0011"], ["mouse_0009", "mouse_0010", "mouse_0011"], ["mouse_0025"]]

Date = ["20201119", "20201125", "20201126", "20201201"]
Mouse = [["mouse_DS_WT_22"], ["mouse_0042", "mouse_0043"], ["mouse_DS_WT_22"], ["mouse_0042", "mouse_0043"]]


Date = ["20201217"]
Mouse = [["time_test"]]

Date = ["20201218"]
Mouse = [["0057"]]

for i_date in range(1, 2):# len(Date)):
    for i_mouse in range(1, 2):# len(Mouse[i_date])):
    
        path = Data + "\\" + Date[i_date] + "\\" + Mouse[i_date][i_mouse]  
        fold = [x for x in os.listdir(path) if isdir(join(path, x))]
        
        
        stim_label = []
        for i_fold in range(0, len(fold)):
            
            words = ["2", "salt", "unregister", "coregister"]
            if not any(ext in fold[i_fold] for ext in words) and i_fold == 4:
                
                stim_label.append(fold[i_fold])
                stim_fold = path + "\\" + fold[i_fold] + "\\FACE"
                onlyfiles = [x for x in os.listdir(stim_fold) if isfile(join(stim_fold, x))  and ".jpg" in x]
                NUM = []
                for i_file in range(0, len(onlyfiles)):
                
                    src = stim_fold + "\\" + onlyfiles[i_file]
                    if isfile(src):
                        
                        numbers = re.findall('[0-9]+', onlyfiles[i_file])
                        NUM.append(int(numbers[-1]))
                
                onlyfiles_sorted = [x for _,x in sorted(zip(NUM,onlyfiles))]
                
                stim_fold = stim_fold + "\\*.jpg"
                
                cnose = FE.findCropCoords(stim_fold)
                cjaw = FE.findCropCoords(stim_fold)
                cear = FE.findCropCoords(stim_fold)
                cpaw = FE.findCropCoords(stim_fold)
                
                mt_nose = []; mt_jaw = []; mt_ear = []; mt_paw = []; imgmin = []
                for m in range(0, len(onlyfiles_sorted)):
                    # img = cv2.imread(Cam + "\\" + camera_rename[l] + "_" + str(NUM[m]) + ".jpg")
                    if ".jpg" in onlyfiles[m]:
                        # img = cv2.imread(stim_fold[0:-5] + onlyfiles_sorted[m])
                        # mt_nose.append(np.mean(img[cnose[1] : cnose[1] + cnose[3], cnose[0] : cnose[0] + cnose[2]]))
                        # mt_jaw.append(np.mean(img[cjaw[1] : cjaw[1] + cjaw[3], cjaw[0] : cjaw[0] + cjaw[2]]))
                        # mt_ear.append(np.mean(img[cear[1] : cear[1] + cear[3], cear[0] : cear[0] + cear[2]]))
                        # mt_paw.append(np.mean(img[cpaw[1] : cpaw[1] + cpaw[3], cpaw[0] : cpaw[0] + cpaw[2]]))
                        
                        imgmin.append(int(onlyfiles_sorted[m][15:19]))
                        
                val = np.unique(imgmin)
                imghist = []
                for i in range(0, len(val)):
                    imghist.append(np.asarray(np.where(np.ravel(imgmin) == val[i])).shape[1])
                    
                # Circular shift values to beginning of run
                idx1 = int(onlyfiles_sorted[0][15:19])
                shift = np.where(np.ravel(val) == idx1)
                shift = shift[0][0]
                # val = np.roll(val, -shift)
                imghist = np.roll(imghist, -shift)
                imghist[0] = 30 # dont count lags from first second, dont know when in second it started
                
                t = np.arange(180,11*120,122)
                t = np.insert(t,0,0)
                # Check total number of images saved - are frames dropped?
                idx_tmp = range(t[0], t[-1])
                lag_tmp = len(idx_tmp)*30 - np.sum(imghist[np.ravel(list(idx_tmp))])
                
                lag = np.zeros(len(t)-1, dtype = int); lag_cum = np.zeros(len(t)-1,dtype = int); 
                if lag_tmp > 10: # more than 1 frame per stimului on average

                    neg = 0
                    for i_trial in range(0, len(t)-1):
                        idx_tmp = range(t[i_trial], t[i_trial + 1])
                        lag_tmp = len(idx_tmp)*30 - np.sum(imghist[np.ravel(list(idx_tmp))])
                        
                        lag[i_trial] = lag_tmp
                        
                        if lag_tmp < -10:
                            neg = i_trial
    
                        lag_cum[i_trial] = np.sum(np.abs(lag[neg:i_trial+1]))
                    
                tt = np.arange(5400,11*3660,3660)
                tt -lag_cum
                
                
                    
                        
                        
                rng = np.random.RandomState(10)  # deterministic random data
                a = np.hstack((rng.normal(size=1000), rng.normal(loc=5, scale=2, size=1000)))         
                _ = plt.hist(imghist, bins='auto')  # arguments are passed to np.histogram
                plt.title("Histogram with 'auto' bins")
                    
                
                f, ax = plt.subplots(1, figsize = (120,10))
                mt_nose2 = pd.Series(mt_nose)
                plt.plot(mt_nose2.rolling(window=10).mean(), linewidth = 10)
                plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
                plt.xlabel('Time (frames)', fontsize = 25)
                plt.ylabel("Pixel Intensity", fontsize = 25)
                plt.title('Nose', fontsize = 25)
                
                f, ax = plt.subplots(1, figsize = (120,10))
                mt_jaw2 = pd.Series(mt_jaw)
                plt.plot(mt_jaw2.rolling(window=10).mean(), linewidth = 10)
                plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
                plt.xlabel('Time (frames)', fontsize = 25)
                plt.ylabel("Pixel Intensity", fontsize = 25)
                plt.title('Jaw', fontsize = 25)
                
                f, ax = plt.subplots(1, figsize = (120,10))
                mt_ear2 = pd.Series(mt_ear)
                plt.plot(mt_ear2.rolling(window=10).mean(), linewidth = 10)
                plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
                plt.xlabel('Time (frames)', fontsize = 25)
                plt.ylabel("Pixel Intensity", fontsize = 25)
                plt.title('Ear', fontsize = 25)
                
                f, ax = plt.subplots(1, figsize = (120,10))
                mt_paw2 = pd.Series(mt_paw)
                plt.plot(mt_paw2.rolling(window=10).mean(), linewidth = 10)
                plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
                plt.xlabel('Time (frames)', fontsize = 25)
                plt.ylabel("Pixel Intensity", fontsize = 25)
                plt.title('Paw', fontsize = 25)
                
                sio.savemat(path + "\\unregister\\mvt_nose.mat", {'mvt_nose':mt_nose})
                sio.savemat(path + "\\unregister\\mvt_jaw.mat", {'mvt_jaw':mt_jaw})
                sio.savemat(path + "\\unregister\\mvt_ear.mat", {'mvt_ear':mt_ear})
                sio.savemat(path + "\\unregister\\mvt_paw.mat", {'mvt_paw':mt_paw})
                
                
                
                
                
                
                
                
                
                
                
                
                
                