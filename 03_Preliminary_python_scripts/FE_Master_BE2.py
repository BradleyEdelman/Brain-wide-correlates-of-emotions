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
from skimage import data, exposure
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
Date = ["20200825", "20200826", "20200827"]
Mouse = [["mouse_0008", "mouse_0009"], ["mouse_0009", "mouse_0010", "mouse_0011"], ["mouse_0009", "mouse_0010", "mouse_0011"]]

for i_date in range(0,1):#, len(Date)):
    for i_mouse in range(0,1):# len(Mouse[i_date])):

        path = Data + "\\" + Date[i_date] + "\\" + Mouse[i_date][i_mouse]
        fold = [x for x in os.listdir(path) if isdir(join(path, x))]
        
        idxn = np.arange(600,20*600,600); idxn = idxn.tolist()
        idxs = np.arange(5400,20*5400,3660)
        idxs = np.tile(idxs, (len(fold)-1, 1)); idxs = idxs.tolist()
        idx = idxs; idx.append(idxn); idx = np.flip(idx)
        
        L = 60 # length of stimulus (frames) (seconds *30)
        B = 120 # length of baseline (frames) (seconds *30)

        # each acquisition run
        proto = []; hogs_all = []; hogs_list = []; hogs_corr = []; hogs_df = [];
        hog_proto = []; hog_proto.append([])
        cnt = []; CC = []; col_colors1 = []
        for i_fold in range(0, 2):# len(fold)):
            
            stim_fold = path + "\\" + fold[i_fold] + "\\FACE"
            onlyfiles = [x for x in os.listdir(stim_fold) if isfile(join(stim_fold, x))  and ".jpg" in x]
            NUM = []
            for i_file in range(0, len(onlyfiles)):
            
                src = stim_fold + "\\" + onlyfiles[i_file]
                if isfile(src):
                    
                    numbers = re.findall('[0-9]+', onlyfiles[i_file])
                    NUM.append(int(numbers[-1]))
            
            onlyfiles_sorted = [x for _,x in sorted(zip(NUM,onlyfiles))]
            img_file = stim_fold + "\\" + onlyfiles[50]
            plt.imshow(mpimg.imread(img_file), cmap = "gray")
            
            stim_fold = stim_fold + "\\*.jpg"
            # Define crop coordinates for all images in folder
            if i_fold == 0:
                
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
            hogs_file = path + "\\" + fold[i_fold] + "_hogs_list.pkl"
            if os.path.isfile(hogs_file):
                with open(hogs_file,"rb") as f:
                   hogs_load = pickle.load(f)
                
                hogs_list.append(hogs_load)
                
            else:
                hogs_list.append(FE.imagesToHogsCellCrop(stim_fold, 32, cropCoords = coord1))
                f = open(hogs_file, 'wb')
                pickle.dump(hogs_list[-1], f)
                
                
            # determine total indices of "stimulation" and "baseline" based on number of trials/files in folder
            count_tmp = len([ii for ii in idx[i_fold] if ii < len(NUM)]); cnt.append(count_tmp - 1)
            IDX_stim = []; IDX_base = []
            for i_cnt in range(0, cnt[-1]):
                IDX_stim = IDX_stim + list(range(idx[i_fold][i_cnt], idx[i_fold][i_cnt] + L))
                IDX_base  = IDX_base + list(range(idx[i_fold][i_cnt] - B, idx[i_fold][i_cnt]))    
            IDX_corr = IDX_stim + IDX_base
                
            hogs_df.append(pd.DataFrame(data = hogs_list[-1]))
            hogs_all.append(hogs_df[-1].iloc[np.r_[IDX_stim]])
            hogs_corr.append(hogs_df[-1].iloc[np.r_[IDX_corr]])   
           
            # pairwise correlation between all frames of interest
            corr = hogs_corr[-1].T.corr()
            #sns.heatmap(corr, robust = True, rasterized = True)
            #g = sns.clustermap(corr, robust=True, rasterized = True)
            
            if fold[i_fold] == 'quinine':
                CC.append('purple')
                CC2 = ['plum', 'mediumorchid', 'blueviolet', 'purple']
            elif fold[i_fold] == 'salt':
                CC.append('orange')
                CC2 = ['bisque', 'sandybrown', 'darkorange', 'saddlebrown']
            elif fold[i_fold] == 'sucrose':
                CC.append('green')
                CC2 = ['lightgreen', 'lime', 'seagreen', 'darkgreen']
            elif fold[i_fold] == 'tail_shock':
                CC.append('red')
                CC2 = ['rosybrown', 'lightcoral', 'red', 'darkred']
            elif fold[i_fold] == 'neutral':
                CC.append('black')
                CC2 = ['lightskyblue','dodgerblue', 'blue', 'navy']
            else:
                CC.append('black')
            col_colors1 = col_colors1 + 60*cnt[-1] * [CC[-1]] # color vector for tSNE
            
            colorVec1 = ([CC[-1]] * 60) * cnt[-1] + (["lightgray"] * 120) * cnt[-1] # color vector for cluster map
            g = sns.clustermap(corr, robust=True, col_colors = colorVec1, row_colors = colorVec1, rasterized = True)  
            g.savefig(path + "\\" + fold[i_fold] + "_cluster.png")  
            
            
            # Look at hogs from neutral and stimulus periods
            if i_fold != 0:
                # Select frames most dissimilar from stimulus frames, keep visualization option
                cluster_order = g.dendrogram_row.reordered_ind
                # obtain just baseline and stimulus frames, as were used in the correlation analysis
                cluster_img = []
                for i_idxcorr in range(0, len(IDX_corr)):
                    cluster_img.append(coll[IDX_corr[i_idxcorr]])
                    
                # find "left-most baseline indices" and "right-most stimulus indices" of dendrogram
                idx_left = []
                for i_idx_left in range(0, len(cluster_order)):
                    if len(idx_left) < 10:
                        if cluster_order[i_idx_left] in list(range(180,540)):
                            idx_left.append(cluster_order[i_idx_left])
                            
                idx_right = []
                for i_idx_right in range(len(cluster_order)-1, 0, -1):
                    if len(idx_right) < 10:
                        if cluster_order[i_idx_right] in list(range(0,180)):
                            idx_right.append(cluster_order[i_idx_right])
                
                idx_sign = [idx_left, idx_right]
                T = ['neutral_' + fold[i_fold], fold[i_fold]]
                for i_sign in range(0, len(idx_sign)):
                    g2 = plt.figure(figsize = (25,10))
                    plt.title(T[i_sign])
                    
                    hog_img_tmp = []; hog_tmp = []
                    for i_num in range(0, len(idx_sign[i_sign])):
                        ax = g2.add_subplot(2,5,i_num+1)
                        
                        img_crop = cluster_img[idx_sign[i_sign][i_num]][coord1[1]:coord1[1]+coord1[3],coord1[0]:coord1[0]+coord1[2]]
                        ax.imshow(img_crop, cmap = 'gray')
                        ax.tick_params(axis = 'x',  which = 'both', bottom = False, top = False, labelbottom = False)
                        ax.tick_params(axis = 'y',  which = 'both', left = False, right = False, labelleft = False)
                        
                        HOG, HOG_img = hog(img_crop, orientations = 8, pixels_per_cell = (32, 32), cells_per_block = (1, 1), visualize = True, transform_sqrt = True)
                        
                        hog_tmp.append(HOG)
                        hog_img_tmp.append(HOG_img)
                        
                    hog_img_ave = np.mean(hog_img_tmp, axis = 0)
                    hog_image_rescaled = exposure.rescale_intensity(hog_img_ave, in_range=(0, .15))
                
                    g2 = plt.figure()
                    ax1 = g2.add_subplot(1,2,1)
                    ax1.axis('off')
                    ax1.imshow(img_crop, cmap=plt.cm.gray)
                    ax1.set_title('Image - ' + T[i_sign])
                    ax2 = g2.add_subplot(1,2,2)
                    ax2.axis('off')
                    ax2.imshow(hog_image_rescaled, cmap = 'gray')
                    ax2.set_title('HoGs - ' + T[i_sign])
                    
                    g2.savefig(path + "\\" + fold[i_fold] + "_HogS_" + T[i_sign] + ".png")
                    
                    if i_sign == 0:
                        hog_proto[0].append(hog_tmp)
                    else:
                        hog_proto.append(hog_tmp)
                     
                    # del hog_tmp, hog_img_tmp


            # pairwise correlation between all frames of interest on an individual trial basis
            for i_cnt in range(0, cnt[-1]):
                
                stim = list(range(idx[i_fold][i_cnt], idx[i_fold][i_cnt] + L))
                base = list(range(idx[i_fold][i_cnt] - B, idx[i_fold][i_cnt]))
                
                hogs_corr2 = hogs_df[-1].iloc[np.r_[stim + base]]
                corr2 = hogs_corr2.T.corr()
                colorVec2 = ([CC2[0]] * 15 + [CC2[1]] * 15 + [CC2[2]] * 15 + [CC2[3]] * 15 + ['lightgray'] * 120)
                g = sns.clustermap(corr2, robust = True, col_colors = colorVec2, row_colors = colorVec2, rasterized = True)
                g.savefig(path + "\\" + fold[i_fold] + "_cluster_trial_" + str(i_cnt+1) + ".png") 
        
            # Creating prototypes
            if fold[i_fold] == "neutral":
                proto.append(hogs_df[-1].mean(axis = 0))
                
                scaler = MinMaxScaler()
                t1 = hogs_df[-1].corrwith(proto[0], axis = 1) # comparing stimulus frames to neutral prototype
                g2 = plt.figure()
                ax = g2.add_subplot()
                ax.plot(scaler.fit_transform(t1.to_numpy().reshape(-1, 1)), color = "black") # neutral face similarity
                protoidx = t1.nsmallest(11).index
                
                g3 = plt.figure(figsize = (25,25))
                for j in range(0, 10):
                    ax = g3.add_subplot(2,5,j+1)
                    ax.imshow(coll[protoidx[j]][coord1[1]:coord1[1]+coord1[3],coord1[0]:coord1[0]+coord1[2]], cmap = 'gray')
                    ax.set_title(str(round(t1[protoidx[j]],2)))
                    ax.tick_params(axis = 'x',  which = 'both', bottom = False, top = False, labelbottom = False)
                    ax.tick_params(axis = 'y',  which = 'both', left = False, right = False, labelleft = False)
                g3.savefig(path + "\\" + fold[i_fold] + "_prototype_faces.png")
                
                
            else:
                
                scaler = MinMaxScaler()
                t1 = hogs_df[-1].corrwith(proto[0], axis = 1) # comparing stimulus frames to neutral prototype
                protoidx = t1.nsmallest(11).index
                proto.append(hogs_df[-1].iloc[protoidx].mean(axis = 0)) # 10 frames most dissimilar from neutral prototype
                t2 = hogs_df[-1].corrwith(proto[-1], axis = 1)
                
                g2 = plt.figure(figsize = (25,25))
                for j in range(0, 10):
                    ax = g2.add_subplot(2,5,j+1)
                    ax.imshow(coll[protoidx[j]][coord1[1]:coord1[1]+coord1[3],coord1[0]:coord1[0]+coord1[2]], cmap = 'gray')
                    ax.set_title(str(round(t1[protoidx[j]],2)))
                    ax.tick_params(axis = 'x',  which = 'both', bottom = False, top = False, labelbottom = False)
                    ax.tick_params(axis = 'y',  which = 'both', left = False, right = False, labelleft = False)
                g2.savefig(path + "\\" + fold[i_fold] + "_prototype_faces.png")
                
                
                g3 = plt.figure()
                ax = g3.add_subplot()
                P = 1000
                t1plot = []; t2plot = []
                for i_cnt in range(0, cnt[-1]):
                    t1plot = t1plot + list(t1[idx[i_fold][i_cnt] - B : idx[i_fold][i_cnt] + P])
                    t2plot = t2plot + list(t2[idx[i_fold][i_cnt] - B : idx[i_fold][i_cnt] + P])
        
                ax.plot(scaler.fit_transform(pd.Series(t1plot).to_numpy().reshape(-1, 1)) - 1, color = "black") # neutral face similarity
                ax.plot(scaler.fit_transform(pd.Series(t2plot).to_numpy().reshape(-1, 1)), color = CC[-1]) # stimulus face similarly
                
                for i_cnt in range(0, cnt[-1]):
                    ax.add_patch(patches.Rectangle((i_cnt * (120 + P),-1),60,2,linewidth = 1, facecolor = 'b', alpha = 0.25))
                    
                g3.savefig(path + "\\" + fold[i_fold] + "_temporal_prototype.png")
                  
                
        # Creating protypes a different way
        P_neut = hog_proto[0][0]
        for i_hog in range(1, len(hog_proto[0])):
            P_neut = np.concatenate([P_neut,hog_proto[0][i_hog]])
        P_neut = pd.DataFrame(data = P_neut)
        P_neut = P_neut.mean(axis = 0)
            
        proto2=[]; proto2.append(P_neut)
        for i_proto in range(1, len(fold)):  
            proto_tmp = hog_proto[i_proto]
            proto_tmp = pd.DataFrame(data = proto_tmp)
            proto_tmp = proto_tmp.mean(axis = 0)
            proto2.append(proto_tmp)
        
        for i_fold in range(1, len(fold)):
            scaler = MinMaxScaler()
            t1 = hogs_df[i_fold].corrwith(proto2[0], axis = 1) # comparing stimulus frames to neutral prototype
            t2 = hogs_df[i_fold].corrwith(proto2[i_fold], axis = 1)
            
            g3 = plt.figure()
            ax = g3.add_subplot()
            P = 1000
            t1plot = []; t2plot = []
            for i_cnt in range(0, cnt[-1]):
                t1plot = t1plot + list(t1[idx[i_fold][i_cnt] - B : idx[i_fold][i_cnt] + P])
                t2plot = t2plot + list(t2[idx[i_fold][i_cnt] - B : idx[i_fold][i_cnt] + P])
    
            ax.plot(scaler.fit_transform(pd.Series(t1plot).to_numpy().reshape(-1, 1)) - 1, color = "black") # neutral face similarity
            ax.plot(scaler.fit_transform(pd.Series(t2plot).to_numpy().reshape(-1, 1)), color = CC[i_fold]) # stimulus face similarly
            
            for i_cnt in range(0, cnt[-1]):
                ax.add_patch(patches.Rectangle((i_cnt * (120 + P),-1),60,2,linewidth = 1, facecolor = 'b', alpha = 0.25))
                
            g3.savefig(path + "\\" + fold[i_fold] + "_temporal_prototype_new.png")

            col_list_palette_pca = sns.xkcd_palette(col_colors1)
            
            #############################################################################################################
            #tSNE_allEmotions variable contains a set of HOGs from a single animal experiencing varius stimuli/emotions.
            pca2 = PCA(n_components=100)
            hogs_all2 = hogs_all[0]
            for i_fold in range(1, len(fold)):
                hogs_all2 = np.vstack([hogs_all2, hogs_all[i_fold]])
            pca2.fit(hogs_all2)
            pcs2 = pca2.fit_transform(hogs_all2)
            tsne2 = TSNE()
            tsne_results2 = tsne2.fit_transform(pcs2)
            g4 = plt.figure()
            plt.scatter(x = tsne_results2[:,0], y = tsne_results2[:,1], color = col_list_palette_pca, alpha = 0.25)
            plt.show()
            g4.savefig(path +  "\\tSNE_projections.png")
            
            
            g5 = plt.figure()
            plt.scatter(x = pcs2[:,0], y = pcs2[:,1], color = col_list_palette_pca, alpha = 0.25)
            g5.savefig(path +  "\\PC_projections.png")
                            
 
            ##############################################################################################################
            # Concatenate all baselines and stimuli for global correlation
            colorVec3 = (["lightgray"] * 60) * cnt[0] + (["lightgray"] * 120) * cnt[0]
            hogs_tmp = hogs_corr[0]
            for i_hog in range(1, len(hogs_corr)):
                hogs_tmp = pd.concat([hogs_tmp, hogs_corr[i_hog]])
                colorVec3 = colorVec3 + ([CC[i_hog]] * 60) * cnt[i_hog] + (["lightgray"] * 120) * cnt[i_hog]
                
            corr3 = hogs_tmp.T.corr()
            g = sns.clustermap(corr3, robust = True, col_colors = colorVec3, row_colors = colorVec3, rasterized = True)
            g.savefig(path + "\\Global_cluster.png")
            
            ##############################################################################################################            
            # Correlate hogs from all stimuli against each prototype
            for i_fold in range(1, len(fold)):

                hogs2 = pd.DataFrame(data = hogs_list[i_fold])
                
                g6 = plt.figure(figsize = (30,5))
                
                t = [];
                for j_fold in range(1, len(fold)):
                    
                    t.append(hogs2.corrwith(proto[j_fold], axis = 1)) # comparing stimulus frames to neutral prototype
                    tmp = list(t[-1])
                    t1plot = [];
                    
                    for i_cnt in range(0, cnt[j_fold]):
                        t1plot = t1plot + list(tmp[idx[j_fold][i_cnt] - B : idx[j_fold][i_cnt] + P])
                       
                    
                    if fold[j_fold] == 'quinine':
                        CCC = 'purple'
                    elif fold[j_fold] == 'salt':
                        CCC = 'orange'
                    elif fold[j_fold] == 'sucrose':
                        CCC = 'green'
                    elif fold[j_fold] == 'tail_shock':
                        CCC = 'red'
                    elif fold[j_fold] == 'neutral':
                        CCC = 'black'
                    else:
                        CCC = 'yellow'
                    
                    ax = g6.add_subplot(1,4,j_fold)
                    if j_fold == i_fold:
                        ax.plot(scaler.fit_transform(pd.Series(t1plot).to_numpy().reshape(-1, 1)), color = CCC, alpha = 0.8)
                    else: 
                        ax.plot(scaler.fit_transform(pd.Series(t1plot).to_numpy().reshape(-1, 1)), color = CCC, alpha = 0.8)
                     
                    AA = 0.5
                    if i_fold == j_fold:
                        AA = 1
                    for i_cnt in range(0, cnt[j_fold]):
                        ax.add_patch(patches.Rectangle((i_cnt * (120 + P),0),60,1,linewidth = 1, facecolor = 'b', alpha = AA))
                        
                    g6.savefig(path + "\\" + fold[i_fold] + "_prototype_vs_all.png")
        



