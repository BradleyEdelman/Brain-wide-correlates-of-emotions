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
from joblib import Parallel, delayed
import cv2
import glob
import re
from sklearn.preprocessing import MinMaxScaler
import os
from os.path import isfile, isdir, join
import FE_helper_func as FE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

# %% 
# example image - full frame

# Specify data path details
multistim = 0 # 1 for yes, 0 for no
Data = r"C:\Users\Bedelman\Videos"
Date = "20200825"
Mouse = "mouse_0009"

path = Data + "\\" + Date + "\\" + Mouse
fold = [x for x in os.listdir(path) if isdir(join(path, x))]

N = [600, 1200, 1800]
S1 = [5400, 9060, 12720] # stimulus start indices
# S1 = [2700, 6360, 10020]
idxn = np.arange(600,20*600,600); idxn = idxn.tolist()
idxs = np.arange(5400,20*5400,3660)
idxs = np.tile(idxs, (len(fold)-1, 1)); idxs = idxs.tolist()
idx = idxs; idx.append(idxn); idx = np.flip(idx)

# S1 = [3200, 7549,10000]
L = 60 # length of stimulus (frames) (seconds *30)
B = 120 # length of baseline (frames) (seconds *30)

# each acquisition run
proto = []; hogs_all = []; hogs_list = [];
col_colors1 = []
for i in range(0, len(fold)):
    
    stim_fold = path + "\\" + fold[i] + "\\FACE"
    onlyfiles = [x for x in os.listdir(stim_fold) if isfile(join(stim_fold, x))  and ".jpg" in x]
    NUM = []
    for j in range(0, len(onlyfiles)):
    
        src = stim_fold + "\\" + onlyfiles[j]
        if isfile(src):
            
            numbers = re.findall('[0-9]+', onlyfiles[j])
            NUM.append(int(numbers[-1]))
    
    onlyfiles_sorted = [x for _,x in sorted(zip(NUM,onlyfiles))]
    img_file = stim_fold + "\\" + onlyfiles[50]
    plt.imshow(mpimg.imread(img_file), cmap = "gray")
    
    stim_fold = stim_fold + "\\*.jpg"
    # Define crop coordinates for all images in folder
    if i == 0:
        
        coord_file = path + "\\crop_coord.pkl"
        if os.path.isfile(coord_file):
            with open(coord_file,"rb") as f:
                coord1 = pickle.load(f)
        else:
            coord1 = FE.findCropCoords(stim_fold)
            f = open(coord_file, 'wb')
            pickle.dump(coord1, f)
      
    # plot uncropped and cropped example image
    coll = io.ImageCollection(stim_fold)
    plt.imshow(coll[50], cmap = 'gray')
    plt.imshow(coll[50][coord1[1]:coord1[1]+coord1[3],coord1[0]:coord1[0]+coord1[2]], cmap = 'gray')
    
    # Extract hogs, save/load hogs, format hogs for various later analyses
    hogs_file = path + "\\" + fold[i] + "_hogs_list.pkl"
    if os.path.isfile(hogs_file):
        with open(hogs_file,"rb") as f:
           hogs_load = pickle.load(f)

        hogs_list.append(hogs_load)
        
    else:
        hogs_list.append(FE.imagesToHogsCellCrop(stim_fold, 32, cropCoords = coord1))
        f = open(hogs_file, 'wb')
        pickle.dump(hogs_list[-1], f)
        
        
    # determine total indices of "stimulation" and "baseline" based on number of trials/files in folder
    count = len([ii for ii in idx[i] if ii < len(NUM)]); count = count - 1
    IDX_stim = []; IDX_base = []
    for j in range(0, count - 1):
        IDX_stim = IDX_stim + list(range(idx[i][j], idx[i][j] + L))
        IDX_base  = IDX_base + list(range(idx[i][j] - B, idx[i][j]))    
        
    hogs_df = pd.DataFrame(data = hogs_list[-1])
    hogs_all = hogs_df.iloc[np.r_[IDX_stim]]
    hogs_corr = hogs_df.iloc[np.r_[IDX_stim + IDX_base]]    
   
    # pairwise correlation between all frames of interest
    corr = hogs_corr.T.corr()
    #sns.heatmap(corr, robust = True, rasterized = True)
    #g = sns.clustermap(corr, robust=True, rasterized = True)
    
    if fold[i] == 'quinine':
        CC = 'purple'
        CC2 = ['plum', 'mediumorchid', 'blueviolet', 'purple']
    elif fold[i] == 'salt':
        CC = 'orange'
        CC2 = ['bisque', 'sandybrown', 'darkorange', 'saddlebrown']
    elif fold[i] == 'sucrose':
        CC = 'green'
        CC2 = ['lightgreen', 'lime', 'seagreen', 'darkgreen']
    elif fold[i] == 'tail_shock':
        CC = 'red'
        CC2 = ['rosybrown', 'lightcoral', 'red', 'darkred']
    elif fold[i] == 'neutral':
        CC = 'black'
        CC2 = ['lightskyblue','dodgerblue', 'blue', 'navy']
    else:
        CC = 'black'
    col_colors1 = col_colors1 + 180 * [CC] # color vector for tSNE
    
    colorVec1 = ([CC] * 60) * count + (["lightgray"] * 120) * count # color vector for cluster map
    g = sns.clustermap(corr, robust=True, col_colors = colorVec1, row_colors = colorVec1, rasterized = True)
        
    g.savefig(path + "\\" + fold[i] + "_cluster.png")    
    
    # pairwise correlation between all frames of interest on an individual trial basis
    g2 = plt.figure(figsize = (25,25))
    for j in range(0, count):
        
        stim = list(range(idx[i][j], idx[i][j] + L))
        base = list(range(idx[i][j] - B, idx[i][j]))
        
        hogs_corr2 = hogs_df.iloc[np.r_[stim + base]]
        corr2 = hogs_corr2.T.corr()
        colorVec2 = ([CC2[0]] * 15 + [CC2[1]] * 15 + [CC2[2]] * 15 + [CC2[3]] * 15 + ['lightgray'] * 120)
        g = sns.clustermap(corr2, robust = True, col_colors = colorVec2, row_colors = colorVec2, rasterized = True)
    

    # Creating prototypes
    if fold[i] == "neutral":
        proto.append(hogs_df.mean(axis = 0))
        
        scaler = MinMaxScaler()
        t1 = hogs_df.corrwith(proto[0], axis = 1) # comparing stimulus frames to neutral prototype
        g3 = plt.figure()
        ax = g3.add_subplot()
        ax.plot(scaler.fit_transform(t1.to_numpy().reshape(-1, 1)), color = "black") # neutral face similarity
        protoidx = t1.nsmallest(11).index
        
        g2 = plt.figure(figsize = (25,25))
        for j in range(0, 10):
            ax = g2.add_subplot(2,5,j+1)
            ax.imshow(coll[protoidx[j]][coord1[1]:coord1[1]+coord1[3],coord1[0]:coord1[0]+coord1[2]], cmap = 'gray')
            ax.set_title(str(round(t1[protoidx[j]],2)))
            ax.tick_params(axis = 'x',  which = 'both', bottom = False, top = False, labelbottom = False)
            ax.tick_params(axis = 'y',  which = 'both', left = False, right = False, labelleft = False)
        g2.savefig(path + "\\" + fold[i] + "_prototype_faces.png")
        
        
    else:
        
        scaler = MinMaxScaler()
        t1 = hogs_df.corrwith(proto[0], axis = 1) # comparing stimulus frames to neutral prototype
        protoidx = t1.nsmallest(11).index
        proto.append(hogs_df.iloc[protoidx].mean(axis = 0)) # 10 frames most dissimilar from neutral prototype
        t2 = hogs_df.corrwith(proto[-1], axis = 1)
        
        g2 = plt.figure(figsize = (25,25))
        for j in range(0, 10):
            ax = g2.add_subplot(2,5,j+1)
            ax.imshow(coll[protoidx[j]][coord1[1]:coord1[1]+coord1[3],coord1[0]:coord1[0]+coord1[2]], cmap = 'gray')
            ax.set_title(str(round(t1[protoidx[j]],2)))
            ax.tick_params(axis = 'x',  which = 'both', bottom = False, top = False, labelbottom = False)
            ax.tick_params(axis = 'y',  which = 'both', left = False, right = False, labelleft = False)
        g2.savefig(path + "\\" + fold[i] + "_prototype_faces.png")
        
        
        g3 = plt.figure()
        ax = g3.add_subplot()
        # t1plot = list(t1[S1[0] - B : S1[0] + 300]) + list(t1[S1[1] - B : S1[1] + 300]) + list(t1[S1[2] - B : S1[2] + 300])
        P = 1000
        t1plot = []; t2plot = []
        for j in range(0,len(S1)):
            t1plot = t1plot + list(t1[S1[j] - B : S1[j] + P])
            t2plot = t2plot + list(t2[S1[j] - B : S1[j] + P])

        ax.plot(scaler.fit_transform(pd.Series(t1plot).to_numpy().reshape(-1, 1)) - 1, color = "black") # neutral face similarity
        ax.plot(scaler.fit_transform(pd.Series(t2plot).to_numpy().reshape(-1, 1)), color = CC) # stimulus face similarly
        
        for j in range(0,len(S1)):
            ax.add_patch(patches.Rectangle((j * (120 + P),-1),60,2,linewidth = 1, facecolor = 'b', alpha = 0.25))
            
        g3.savefig(path + "\\" + fold[i] + "_temporal_prototype.png")

        
# %%
        
for i in range(1, len(fold)):

    hogs2 = pd.DataFrame(data = hogs_list[i])
    
    g5 = plt.figure()
    ax = g5.add_subplot()
    
    t = [];
    for j in range(1, len(fold)):
        
        t.append(hogs2.corrwith(proto[j], axis = 1)) # comparing stimulus frames to neutral prototype
        tmp = list(t[-1])
        t1plot = [];
        
        if i == 0:
            STIM = N
        else:
            STIM = S1
        for k in range(0, len(STIM)):
            t1plot = t1plot + list(tmp[STIM[k] - B : STIM[k] + P])
        
        if fold[j] == 'quinine':
            CCC = 'purple'
        elif fold[j] == 'salt':
            CCC = 'orange'
        elif fold[j] == 'sucrose':
            CCC = 'green'
        elif fold[j] == 'tail_shock':
            CCC = 'red'
        elif fold[j] == 'neutral':
            CCC = 'black'
        else:
            CCC = 'yellow'
        
        if j == i:
            ax.plot(scaler.fit_transform(pd.Series(t1plot).to_numpy().reshape(-1, 1)), color = CCC)
        else: 
            ax.plot(scaler.fit_transform(pd.Series(t1plot).to_numpy().reshape(-1, 1)), color = CCC, alpha = 0.5)
            
        for j in range(0,len(S1)):
            ax.add_patch(patches.Rectangle((j * (120 + P),-1),60,2,linewidth = 1, facecolor = 'b', alpha = 0.25))
            
# %%        
col_list_palette_pca = sns.xkcd_palette(col_colors1)

#tSNE_allEmotions variable contains a set of HOGs from a single animal experiencing varius stimuli/emotions.

#t-SNE reduces dimensions of each HOG to 2, which are plotted on a scatter plot and post-hoc
#colored based on the stimulus presented.
#Before running the t-SNE algorithm, number of dimensions are reduced to 100 using principal component analysis.
#gray = baseline, purple = bitter, green = sweet, gold/yellow = malaise, red = pain, dark blue = passive fear, light blue = active fear
pca2 = PCA(n_components=100)
pca2.fit(hogs_all)
pcs2 = pca2.fit_transform(hogs_all)
tsne2 = TSNE()
tsne_results2 = tsne2.fit_transform(pcs2)
g4 = plt.figure()
plt.scatter(x = tsne_results2[:,0], y = tsne_results2[:,1], color = col_list_palette_pca, alpha = 0.25)
plt.show()
g4.savefig(path +  "\\tSNE_projections.png")


g5 = plt.figure()
plt.scatter(x = pcs2[:,0], y = pcs2[:,1], color = col_list_palette_pca, alpha = 0.25)
g5.savefig(path +  "\\PC_projections.png")

