# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:17:03 2020

@author: bedelman
"""

import pandas as pd
import seaborn as sns
import numpy as np  
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import matplotlib.patches as patches
from skimage import io
from skimage.feature import hog
import re
from sklearn.preprocessing import MinMaxScaler
import os
from os.path import isfile, isdir, join
import FE_helper as FE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

# %% 
# example image - full frame

# Specify data path details
multistim = 0 # 1 for yes, 0 for no
Data = r"J:\Bradley Edelman\Facial_Exp_fUS"
Data_HOGS = Data + "_HOGS"

Date = ["20210510", "20210511", "20210512", "20210513", "20210514", "20210515"]
Mouse = ["mouse_0119", "mouse_0120"]

Date = ["20210411", "20210412", "20210413", "20210414", "20210415", "20210416"]
Mouse = ["mouse_0070", "mouse_0071", "mouse_0072", "mouse_0073"]

rewrite = 1
register = 1

pix_cell = 24
for i_mouse in range(1, 2):#len(Mouse)):
    
    fold =[]; fold_analysis = []
    # find all folders for this mouse, across specified days
    for i_date in range(0, len(Date)):
        
        path = Data + "\\" + Date[i_date] + "\\" + Mouse[i_mouse]
        
        # create analysis path
        path_analysis = Data_HOGS + "\\" + Date[i_date]
        if not os.path.isdir(path_analysis):
            os.mkdir(path_analysis, 0o777)
        path_analysis = path_analysis + "\\" + Mouse[i_mouse]
        if not os.path.isdir(path_analysis):
            os.mkdir(path_analysis, 0o777)
        
        # extract stimulus folders and ensure analysis path exists
        stim = [x for x in os.listdir(path) if isdir(join(path, x))]
        for i_stim in range(0, len(stim)):
            fold.append(path + "\\" + stim[i_stim] + "\\FACE")
            
            fold_analysis_tmp = path_analysis + "\\" + stim[i_stim]
            if not os.path.isdir(fold_analysis_tmp):
                os.mkdir(fold_analysis_tmp, 0o777)
            
            fold_analysis.append(fold_analysis_tmp + "\\FACE")
            if not os.path.isdir(fold_analysis[-1]):
                os.mkdir(fold_analysis[-1], 0o777)
                
            
    for i_fold in range(0, 1):#len(fold)):
        
        hogs_file = fold_analysis[i_fold] + "\\hogs_list.pkl"
        
        # if hogs file doesnt exist or want to rewrite it
        if not os.path.isfile(hogs_file) or rewrite == 1:
            
            # if day 1, define, apply crop coords and create hogs
            coord_file = fold_analysis[0] + "\\crop_coord.pkl"
            if i_fold == 0:
                
                if os.path.isfile(coord_file):
                    with open(coord_file,"rb") as f:
                        cropCoords = pickle.load(f)
                else:
                    cropCoords = FE.findCropCoords(fold[i_fold] + "\\*.jpg")
                    f = open(coord_file, 'wb')
                    pickle.dump(cropCoords, f)
                
                hogs = FE.imagesToHogsCellCrop(fold[0] + "\\*.jpg", pix_cell, cropCoords)
            
            # also define spout coordinates to "zero-out" later
            spout_file = fold_analysis[0] + "\\spout_crop_coord.pkl"
            if i_fold == 0:
                
                if not os.path.isfile(spout_file):
                    coords2, coords3, idxz, DIM = FE.findCropCoords2(fold[i_fold] + "\\*.jpg", cropCoords, pix_cell, 8)
                    f = open(spout_file, 'wb')
                    pickle.dump([coords2, coords3, idxz, DIM], f)
                    
            # if session 2+, register to session 1, apply crop coords and create hogs    
            elif i_fold > 0 and os.path.isfile(coord_file):
                
                with open(coord_file,"rb") as f:
                    cropCoords = pickle.load(f)
                
                FE.visAlignFolder(fold[0] + "\\*.jpg", fold[i_fold] + "\\*.jpg")   
                hogs = FE.imagesToHogsCellCropAlignFolder(fold[0] + "\\*.jpg", fold[i_fold] + "\\*.jpg", pix_cell, cropCoords)
                
            if hogs: 
                f = open(hogs_file, 'wb')
                pickle.dump(hogs, f)
            