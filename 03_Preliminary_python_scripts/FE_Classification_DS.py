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
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import random

# %% 
# example image - full frame

# Specify data path details
Data = r"C:\Users\Bedelman\Videos\Facial_Exp"

Date = ["20201119", "20201125", "20201126", "20201201"]
Mouse = [["mouse_DS_WT_22"], ["mouse_0042", "mouse_0043"], ["mouse_DS_WT_22"], ["mouse_0042", "mouse_0043"]]


for i_date in range(0, len(Date)):
    for i_mouse in range(0, len(Mouse[i_date])):

        path = Data + "\\" + Date[i_date] + "\\" + Mouse[i_date][i_mouse]
        # create results folder
        results_path = path + "\\results\\"
        if not os.path.exists(path):
            os.makedirs(path)
        
        # find run folders in mouse session folders  
        fold = [x for x in os.listdir(path) if isdir(join(path, x))]
        
        idxn = np.arange(600,20*600,600); # indices of neutral periods
        idxs = np.arange(5400,20*5400,3660) # indices of stimulus periods
        idx = []; idx.append(idxn)
        for ii in range(0, len(fold) - 1):
            idx.append(idxs)
        
        L = 60 # length of stimulus (frames) (seconds *30)
        B = 120 # length of baseline (frames) (seconds *30)

        # each acquisition run
        proto = []; proto_img = [];
        hogs_all = []; hogs_list = []; hogs_corr = []; hogs_df = [];
        cnt = []; CC = []; col_colors1 = []
        stim_label = []
        
        pix_cell = 32; orient = 8 # some HOG parameters
        IDXX = 0 
        for i_fold in range(0, len(fold)):
            
            words = ["data"] # Do no analyze folders containing these words
            if not any(ext in fold[i_fold] for ext in words):
            
                # stim_label.append(fold[i_fold])
                # Specific camera folder
                stim_fold = path + "\\" + fold[i_fold] + "\\FACE"
                # Find all .jpgs
                onlyfiles = [x for x in os.listdir(stim_fold) if isfile(join(stim_fold, x))  and ".jpg" in x]
                NUM = []
                for i_file in range(0, len(onlyfiles)):
                
                    src = stim_fold + "\\" + onlyfiles[i_file]
                    if isfile(src):
                        
                        numbers = re.findall('[0-9]+', onlyfiles[i_file])
                        NUM.append(int(numbers[-1]))
                
                # Make sure .jpg file list is in correct order
                onlyfiles_sorted = [x for _,x in sorted(zip(NUM,onlyfiles))]
                
                # Define crop coordinates for all images in folder 
                stim_fold = stim_fold + "\\*.jpg"
                coord_file = results_path + "crop_coord.pkl"
                # load pre-existing coordinate file
                if os.path.isfile(coord_file):
                    with open(coord_file,"rb") as f:
                        coord = pickle.load(f)
                else:
                    # or re-define and save coordinates
                    coord = FE.findCropCoords(stim_fold)
                    # want to crop full width of image no matter what?
                    coord = list(coord); coord[0] = 0; coord[2] = 1280;
                    f = open(coord_file, 'wb')
                    pickle.dump(coord, f)
                        
                
                # Extract hogs, save/load hogs, format hogs for later analyses
                hogs_file = results_path + fold[i_fold] + "_hogs_list.pkl"
                if os.path.isfile(hogs_file):
                    with open(hogs_file,"rb") as f:
                       hogs_load = pickle.load(f)
                    
                    hogs_list.append(hogs_load)
                    
                else:
                    hogs_list.append(FE.imagesToHogsCellCrop(stim_fold, pix_cell, cropCoords = coord))
                    f = open(hogs_file, 'wb')
                    pickle.dump(hogs_list[-1], f)
                    
                    
                # determine total indices of "stimulation" and "baseline" based on number of trials/files in folder
                count_tmp = len([ii for ii in idx[IDXX] if ii < len(NUM)]); cnt.append(count_tmp - 1)
                IDX_stim = []; IDX_base = []
                for i_cnt in range(0, cnt[-1]):
                    IDX_stim = IDX_stim + list(range(idx[IDXX][i_cnt], idx[IDXX][i_cnt] + L))
                    IDX_base  = IDX_base + list(range(idx[IDXX][i_cnt] - B, idx[IDXX][i_cnt]))    
                IDX_corr = IDX_stim + IDX_base
                    
                hogs_df.append(pd.DataFrame(data = hogs_list[-1])) # Total hogs
                hogs_all.append(hogs_df[-1].iloc[np.r_[IDX_stim]]) # Stimuluation hogs
                hogs_corr.append(hogs_df[-1].iloc[np.r_[IDX_corr]]) # Stimulation and baseline hogs
            
                # Create color scheme for different stimuli
                if fold[i_fold] == 'quinine':
                    CC.append('purple')
                elif fold[i_fold] == 'salt':
                    CC.append('orange')
                elif fold[i_fold] == 'sucrose':
                    CC.append('green')
                elif fold[i_fold] == 'tail_shock':
                    CC.append('red')
                elif fold[i_fold] == 'neutral':
                    CC.append('black')
                else:
                    CC.append('black')
                col_colors1 = col_colors1 + L*cnt[-1] * [CC[-1]] # color vector for tSNE
                
            # sometimes need to skip some folders
            if "_2" in fold[i_fold] and "r_2" not in fold[i_fold]:
                IDXX = IDXX + 1

# %%    
        # PCA
        
        # color palette for pca plotting
        col_list_palette_pca = sns.xkcd_palette(col_colors1)
        # Extract top 100 PCs
        pca2 = PCA(n_components=100)
        # Concatenate all stimulus hog frames
        hogs_all2 = hogs_all[0]
        for i_fold in range(1, len(hogs_all)):
            hogs_all2 = np.vstack([hogs_all2, hogs_all[i_fold]])
        pca2.fit(hogs_all2)
        
        PCA_model_file = results_path + "PCA_model.pkl"
        f = open(PCA_model_file, 'wb')
        pickle.dump(pca2, f)  
        
        # Determine PC loadings for data projection
        loadings = pca2.components_.T * np.sqrt(pca2.explained_variance_)
        loading_matrix = pd.DataFrame(loadings)
        load = []
        for i_pc in range(0, loadings.shape[1]):
            load.append(np.matmul(hogs_all2, loading_matrix[i_pc].to_numpy()))
        
        loading_matrix = pd.DataFrame(loadings)
        PCA_loading_file = results_path + "PCA_loadings.pkl"
        f = open(PCA_loading_file, 'wb')
        pickle.dump(loading_matrix, f)
        
        # plot first two PCs
        g = plt.figure()
        plt.scatter(load[0], load[1], c = np.asarray(col_list_palette_pca), alpha = 0.5)
        plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('PC Projections')
        plt.xlim(-12,12); plt.ylim(-12,12)
        g.savefig(results_path + "Native_PC_projections.png")
        
        # Determine and plot variance explained by extracted PCs
        X_train = FE.explained_variance(pca2, hogs_all2)
        g = plt.figure()
        plt.plot(np.cumsum(X_train))
        plt.xlabel('# of PCs'); plt.ylabel('Variance Explained (a.u.)')
        plt.ylim(0, 1)
        plt.title('PC Variance Explained')
        g.savefig(results_path + "PC_Variance_Explained.png")
        
        # Create example hog to get resulting dimensions
        img = cv2.imread(glob.glob(stim_fold)[0],0)
        img_crop = img[coord[1]:coord[1]+coord[3],coord[0]:coord[0]+coord[2]]
        HOG, img_hog = hog(img_crop, orientations = orient, pixels_per_cell = (pix_cell, pix_cell), cells_per_block = (1, 1), visualize = True, transform_sqrt = True)
                
        DIM = img_hog.shape
        DIM = np.floor(np.divide(DIM,pix_cell))
        DIM = DIM.astype(int)
        tot_pix = DIM[0]*DIM[1]
        
        # PC visualization
        for i_pc in range(0, 2):
            g = plt.figure(figsize = (10, 3))
            pc_tmp = []
            for i_orient in range(0, 8):
                pc_tmp.append(loading_matrix[i_pc])
                # need to visualize all 8 orientations
                pc_tmp[-1] = pc_tmp[-1][i_orient:len(hogs_list[-1][1]):8]
                pc_tmp[-1] = np.reshape(np.ravel(pc_tmp[-1]), (int(DIM[0]), int(DIM[1])))
                ax = plt.subplot(2, 4, ii+1)
                ax.imshow(pc_tmp[-1])
                ax.axis('off')
                ax.set_title("orientation: " + str(i_orient+1))
                
            g.savefig(results_path +  "PC_" + str(i_pc+1) + "_orientations.png")
            
            # Also visualize  absolute PC magnitude, averaged across orientations 
            g5 = plt.figure()
            plt.imshow(np.mean(np.absolute(pc_tmp), axis = 0))
            plt.axis('off')
            plt.title("average")
            g5.savefig(results_path +  "PC_" + str(i_pc+1) + "_average.png")
# %%    
        # Classification (This is a bit hardcoded at the moment)
        
        all_data = pd.concat([hogs_all[0], hogs_all[1], hogs_all[2]])
        
        # project hog data onto first two principle components
        pc_data1 = pd.DataFrame(np.matmul(all_data.to_numpy(),loading_matrix[0].to_numpy()))
        pc_data2 = pd.DataFrame(np.matmul(all_data.to_numpy(),loading_matrix[1].to_numpy()))
        pc_data3 = pd.concat([pc_data1, pc_data2], axis = 1)
        
        # 100-fold cross validation
        acc = []; acc_n = []; acc_s =[]; acc_ts = []
        for i_iter in range(0, 100):
            print(i_iter)
            # select equal PERCENTAGE of training samples from each class
            class_idx = []
            class_idx.append(list(range(0, len(hogs_all[0]))))
            class_idx.append(list(range(len(hogs_all[0]), len(hogs_all[0])+len(hogs_all[1]))))
            class_idx.append(list(range(len(hogs_all[1]), len(hogs_all[1])+len(hogs_all[2]))))                
            
            # combine all train indices
            train_idx = []
            for i in range(0,3):
                train_idx = train_idx + random.sample(class_idx[i], k = int(round(len(class_idx[i])*0.15)))
            
            # select remaining samples for testing
            class_idx_tot = class_idx[0] + class_idx[1] + class_idx[2]
            test_idx = [x for x in class_idx_tot if x not in train_idx]
            
            # Label data
            colorVecClass = 120 * ['gray'] + 600 * ['green'] + 600 * ['red']
            df_col_colors1 = pd.DataFrame(colorVecClass)
            train_labels = df_col_colors1.iloc[list(train_idx)]
            test_labels = df_col_colors1.iloc[list(test_idx)]
            
            train_data = pc_data3.iloc[train_idx]
            test_data = pc_data3.iloc[test_idx]
            
            # Establish classifier
            rf_1 = RandomForestClassifier()
            # Train classifier
            rf_1.fit(train_data, np.ravel(train_labels))
            # Predict unseen data
            predictions_1 = rf_1.predict(test_data)
            
            # total and individual class accuracy values
            acc.append(metrics.accuracy_score(test_labels, predictions_1))
            acc_n.append(metrics.accuracy_score(test_labels[0:len(hogs_all[0])], predictions_1[0:len(hogs_all[0])]))
            acc_s.append(metrics.accuracy_score(test_labels[len(hogs_all[0]):len(hogs_all[0])+len(hogs_all[1])], predictions_1[len(hogs_all[0]):len(hogs_all[0])+len(hogs_all[1])]))
            acc_ts.append(metrics.accuracy_score(test_labels[len(hogs_all[1]):len(hogs_all[1])+len(hogs_all[2])], predictions_1[len(hogs_all[1]):len(hogs_all[1])+len(hogs_all[2])]))
            
        classification_file = results_path + "classification.pkl"
        f = open(classification_file, 'wb')
        pickle.dump([acc, acc_n, acc_s, acc_ts], f)
        
        # plot x-fold classification results
        g = plt.figure()
        M = [np.mean(acc_n), np.mean(acc_s), np.mean(acc_ts), np.mean(acc)]
        STD = [np.std(acc_n), np.std(acc_s), np.std(acc_ts), np.std(acc)]
        C = ['gray', 'green', 'red', 'black']
        L = ['neutral', 'sucrose', 'tail_shock', 'total']
        
        plt.bar([1, 2, 3, 4], M, yerr = STD, color = C, width = 0.8, tick_label = L)
        plt.ylabel('Classification Accuracy (%)'); plt.ylim(0,1.1)
        plt.title('PC Projection')
        g.savefig(path + "Classification.png")
        




