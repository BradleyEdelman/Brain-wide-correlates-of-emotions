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
from sklearn.preprocessing import MinMaxScaler
import os
from os.path import isfile, isdir, join
import FE_helper_func as FE
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# %% 
# example image - full frame

# Specify data path details
multistim = 0 # 1 for yes, 0 for no
Data = r"R:\Work - Experiments\Data\Brad\Facial_Expression_Videos"
Date = "20200827"
Mouse = "mouse_0009"

path = Data + "\\" + Date + "\\" + Mouse
fold = os.listdir(path)

# each acquisition run
proto = []
hogs_all = []
col_colors1 = []
for i in range(0, len(fold)):
    
    stim_fold = path + "\\" + fold[i] + "\\face"
    onlyfiles = [x for x in os.listdir(stim_fold) if isfile(join(stim_fold, x))]
        
    img_file = stim_fold + "\\" + onlyfiles[50]
    plt.imshow(mpimg.imread(img_file), cmap = "gray")
    
    data_fold = stim_fold + "\\*.jpg"
    
    if i == 0:
        # Define crop coordinates for all images in folder
        
        coord1 = FE.findCropCoords(data_fold)
      
    # plot uncropped and cropped example image
    coll = io.ImageCollection(data_fold)
    plt.imshow(coll[50], cmap = 'gray')
    plt.imshow(coll[50][coord1[1]:coord1[1]+coord1[3],coord1[0]:coord1[0]+coord1[2]], cmap = 'gray')

    # Crop images and extract HOG features
    hogs = FE.imagesToHogsCellCrop(data_fold, 32, cropCoords = coord1)
    hogs = pd.DataFrame(data = hogs)
    
    # Extract hogs during only stimuli for later tSNE analysis
    if i == 0:
        hogs_all = hogs.iloc[np.r_[120:180, 300:360, 480:540]]
    else:
        hogs_all = np.concatenate((hogs_all, hogs.iloc[np.r_[120:180, 300:360, 480:540]]), axis = 0)

    # pairwise correlation between all frames of interest
    corr = hogs.T.corr()
    #sns.heatmap(corr, robust = True, rasterized = True)
    #g = sns.clustermap(corr, robust=True, rasterized = True)
    
    if fold[i] == 'quinine':
        CC = 'purple'
    elif fold[i] == 'salt':
        CC = 'orange'
    elif fold[i] == 'sucrose':
        CC = 'green'
    elif fold[i] == 'shock':
        CC = 'red'
    elif fold[i] == 'neutral':
        CC = 'black'
    col_colors1 = col_colors1 + 180 * [CC] # color vector for tSNE
        
    colorVec1 = (["lightgray"] * 120 + [CC] * 60) * 3 # color vector for cluster map
    g = sns.clustermap(corr, robust=True, col_colors = colorVec1, row_colors = colorVec1, rasterized = True)
        
    g.savefig(data_fold[0:-5] + fold[i] + "_cluster.png")    
    
    # Creating prototypes
    if fold[i] == "neutral":
        proto.append(hogs.mean(axis = 0))
        
        scaler = MinMaxScaler()
        t1 = hogs.corrwith(proto[0], axis = 1) # comparing stimulus frames to neutral prototype
        g3 = plt.figure()
        ax = g3.add_subplot()
        ax.plot(scaler.fit_transform(t1.to_numpy().reshape(-1, 1)), color = "black") # neutral face similarity
        protoidx = t1.nsmallest(11).index
        
        g2 = plt.figure(figsize = (25,25))
        for idx in range(0, 10):
            ax = g2.add_subplot(2,5,idx+1)
            ax.imshow(coll[protoidx[idx]][coord1[1]:coord1[1]+coord1[3],coord1[0]:coord1[0]+coord1[2]], cmap = 'gray')
            ax.set_title(str(round(t1[protoidx[idx]],2)))
            ax.tick_params(axis = 'x',  which = 'both', bottom = False, top = False, labelbottom = False)
            ax.tick_params(axis = 'y',  which = 'both', left = False, right = False, labelleft = False)
        g2.savefig(data_fold[0:-5] + fold[i] + "_prototype_faces.png")
        
        
    else:
        
        scaler = MinMaxScaler()
        t1 = hogs.corrwith(proto[0], axis = 1) # comparing stimulus frames to neutral prototype
        protoidx = t1.nsmallest(11).index
        proto.append(hogs.iloc[protoidx].mean(axis = 0)) # 10 frames most dissimilar from neutral prototype
        t2 = hogs.corrwith(proto[-1], axis = 1)
        
        g2 = plt.figure(figsize = (25,25))
        for idx in range(0, 10):
            ax = g2.add_subplot(2,5,idx+1)
            ax.imshow(coll[protoidx[idx]][coord1[1]:coord1[1]+coord1[3],coord1[0]:coord1[0]+coord1[2]], cmap = 'gray')
            ax.set_title(str(round(t1[protoidx[idx]],2)))
            ax.tick_params(axis = 'x',  which = 'both', bottom = False, top = False, labelbottom = False)
            ax.tick_params(axis = 'y',  which = 'both', left = False, right = False, labelleft = False)
        g2.savefig(data_fold[0:-5] + fold[i] + "_prototype_faces.png")
        
        
        g3 = plt.figure()
        ax = g3.add_subplot()
        ax.plot(scaler.fit_transform(t1.to_numpy().reshape(-1, 1)) - 1, color = "black") # neutral face similarity
        ax.plot(scaler.fit_transform(t2.to_numpy().reshape(-1, 1)), color = CC) # stimulus face similarly
        ax.add_patch(patches.Rectangle((120,-1),60,2,linewidth=1,facecolor='b', alpha = 0.25))
        ax.add_patch(patches.Rectangle((300,-1),60,2,linewidth=1,facecolor='b', alpha = 0.25))
        ax.add_patch(patches.Rectangle((480,-1),60,2,linewidth=1,facecolor='b', alpha = 0.25))
        g3.savefig(data_fold[0:-5] + fold[i] + "_temporal_prototype.png")
        
        
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
g4.savefig(path +  "_tSNE_projections.png")



