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
multistim = 0 # 1 for yes, 0 for no
Data = r"C:\Users\Bedelman\Videos\Facial_Exp"
# Date = ["sample", "20200825", "20200826", "20200827", "20200923"]
# Mouse = [["mouse_0000", "mouse_0001"],["mouse_0008 - Copy", "mouse_0009"], ["mouse_0009", "mouse_0010", "mouse_0011"], ["mouse_0009", "mouse_0010", "mouse_0011"], ["mouse_0025"]]

Date = ["20201119", "20201125", "20201126", "20201201"]
Mouse = [["mouse_DS_WT_22"], ["mouse_0042", "mouse_0043"], ["mouse_DS_WT_22"], ["mouse_0042", "mouse_0043"]]


for i_date in range(1, 2):# len(Date)):
    for i_mouse in range(0, 1):# len(Mouse[i_date])):

        path = Data + "\\" + Date[i_date] + "\\" + Mouse[i_date][i_mouse]
        reg_path = path + "\\unregister\\"
        if not os.path.exists(reg_path):
            os.makedirs(reg_path)
            
        fold = [x for x in os.listdir(path) if isdir(join(path, x))]
        
        idxn = np.arange(600,20*600,600); # idxn = idxn.tolist()
        idxs = np.arange(5400,20*5400,3660)
        # idxs = np.arange(3600,20*3600,3660)
        # idxn = np.arange(50,20*100,100); idxn = idxn.tolist()
        # idxs = np.arange(50,20*100,100)
        idx = []; idx.append(idxn)
        for ii in range(0, len(fold) - 1):
            idx.append(idxs)
        
        L = 60 # length of stimulus (frames) (seconds *30)
        B = 120 # length of baseline (frames) (seconds *30)
        # L = 60
        # B=50

        # each acquisition run
        proto = []; proto_img = [];
        hogs_all = []; hogs_list = []; hogs_corr = []; hogs_df = [];
        cnt = []; CC = []; col_colors1 = []
        stim_label = []
        
        pix_cell = 32
        orient = 8
        IDXX = 0 
        for i_fold in range(0, len(fold)):
            
            words = ["2", "salt", "unregister", "coregister"]
            if not any(ext in fold[i_fold] for ext in words):
            
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
                # Define crop coordinates for all images in folder
                if fold[i_fold] == 'neutral':
                    
                    coord_file = reg_path + "crop_coord.pkl"
                    if os.path.isfile(coord_file):
                        with open(coord_file,"rb") as f:
                            coord1 = pickle.load(f)
                    else:
                        coord1 = FE.findCropCoords(stim_fold)
                        coord1 = list(coord1); coord1[0] = 0; coord1[2] = 1280;
                        f = open(coord_file, 'wb')
                        pickle.dump(coord1, f)
                        
                    spout_img_coord_file = reg_path + "spout_img_crop_coord.pkl"    
                    spout_hog_coord_file = reg_path + "spout_hog_crop_coord.pkl"
                    spout_hog_idx_file = reg_path + "spout_hog_crop_idx.pkl"
                    spout_DIM_file = reg_path + "spout_crop_DIM.pkl"
                    if os.path.isfile(spout_img_coord_file) & os.path.isfile(spout_hog_coord_file) & os.path.isfile(spout_hog_idx_file) & os.path.isfile(spout_DIM_file):
                        with open(spout_img_coord_file,"rb") as ff:
                            spout_img_coord = pickle.load(ff)
                            
                        with open(spout_hog_coord_file,"rb") as ff:
                            spout_hog_coord = pickle.load(ff)
                            
                        with open(spout_hog_idx_file,"rb") as ff:
                            spout_idx = pickle.load(ff)
                            
                        with open(spout_DIM_file,"rb") as ff:
                            spout_DIM = pickle.load(ff)
                            
                    else:
                        spout_img_coord, spout_hog_coord, spout_idx, spout_DIM = FE.findCropCoords2(path + "\\sucrose\\FACE\\*.jpg", coord1, pix_cell, orient) 
                        ff = open(spout_img_coord_file, 'wb')
                        pickle.dump(spout_img_coord, ff)
                        
                        ff = open(spout_hog_coord_file, 'wb')
                        pickle.dump(spout_hog_coord, ff)
                        
                        ff = open(spout_hog_idx_file, 'wb')
                        pickle.dump(spout_idx, ff)
                        
                        ff = open(spout_DIM_file, 'wb')
                        pickle.dump(spout_DIM, ff)
                        
                
                # Extract hogs, save/load hogs, format hogs for various later analyses
                hogs_file = reg_path + fold[i_fold] + "_hogs_list.pkl"
                if os.path.isfile(hogs_file):
                    with open(hogs_file,"rb") as f:
                       hogs_load = pickle.load(f)
                    
                    hogs_list.append(hogs_load)
                    
                else:
                    hogs_list.append(FE.imagesToHogsCellCrop(stim_fold, pix_cell, cropCoords = coord1))
                    f = open(hogs_file, 'wb')
                    pickle.dump(hogs_list[-1], f)
                    
                    
                # zero out all spout indices in hog
                for i_hog in range(0,len(hogs_list[-1])):
                    hogs_list[-1][i_hog][spout_idx] = 0
                    
                g5 = plt.figure() # check spout removal
                for ii in range(0, 8):
                    pc1_tmp = hogs_list[-1][2]
                    pc1_tmp = pc1_tmp[ii:len(hogs_list[-1][1]):8]
                    pc1_tmp = np.reshape(pc1_tmp, (int(spout_DIM[0]), int(spout_DIM[1])))
                    # pc1_tmp[int(coords3[1]):int(coords3[1])+int(coords3[3]),int(coords3[0]):int(coords3[0])+int(coords3[2])] = 0
                    ax = plt.subplot(2, 4, ii+1)
                    ax.imshow(pc1_tmp)
                    
                    
                # determine total indices of "stimulation" and "baseline" based on number of trials/files in folder
                count_tmp = len([ii for ii in idx[IDXX] if ii < len(NUM)]); cnt.append(count_tmp - 1)
                IDX_stim = []; IDX_base = []
                for i_cnt in range(0, cnt[-1]):
                    IDX_stim = IDX_stim + list(range(idx[IDXX][i_cnt], idx[IDXX][i_cnt] + L))
                    IDX_base  = IDX_base + list(range(idx[IDXX][i_cnt] - B, idx[IDXX][i_cnt]))    
                IDX_corr = IDX_stim + IDX_base
                    
                hogs_df.append(pd.DataFrame(data = hogs_list[-1]))
                hogs_all.append(hogs_df[-1].iloc[np.r_[IDX_stim]])
                hogs_corr.append(hogs_df[-1].iloc[np.r_[IDX_corr]])
            
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
                
                cluster_file = reg_path + fold[i_fold] + "_cluster.png"
                colorVec1 = ([CC[-1]] * L) * cnt[-1] + (["lightgray"] * B) * cnt[-1] # color vector for cluster map
                if not isfile(cluster_file):
                    # pairwise correlation between all frames of interest
                    corr = hogs_corr[-1].T.corr()
                    #sns.heatmap(corr, robust = True, rasterized = True)
                    #g = sns.clustermap(corr, robust=True, rasterized = True)
                    g = sns.clustermap(corr, robust=True, col_colors = colorVec1, row_colors = colorVec1, rasterized = True)  
                    g.savefig(reg_path + fold[i_fold] + "_cluster.png")
                
                # find selected images to create prototype
                proto_fold = path + "\\" + fold[i_fold] + "_2\\*.jpg"
                # plot uncropped and cropped example image
                # proto_fold = path + "\\" + fold[i_fold] + "\\Face\\*.jpg"
                coll = io.ImageCollection(proto_fold)
                img = coll[-1]
                # III = 15
                # img = coll[III]
                img_crop = img[coord1[1]:coord1[1]+coord1[3],coord1[0]:coord1[0]+coord1[2]]
                HOG, img_hog = hog(img_crop, orientations = 8, pixels_per_cell = (pix_cell, pix_cell), cells_per_block = (1, 1), visualize = True, transform_sqrt = True)
                img_hog_rescale = exposure.rescale_intensity(img_hog, in_range=(0, .15))\
                
                g2 = plt.figure(figsize = (25,10))
                ax1 = plt.subplot(1,3,1)
                ax1.axis('off')
                ax1.imshow(img, cmap = plt.cm.gray)
                ax1.set_title(fold[i_fold])
                ax2 = plt.subplot(1,3,2)
                ax2.axis('off')
                ax2.imshow(img_crop, cmap = plt.cm.gray)
                ax2.set_title(fold[i_fold] + ' - cropped')
                ax3 = plt.subplot(1,3,3)
                ax3.axis('off')
                ax3.imshow(img_hog_rescale, cmap = plt.cm.gray)
                ax3.set_title(fold[i_fold] + ' - HoG')
                g2.savefig(path + "\\" + fold[i_fold] + "_HoG.png")
                plt.show()
                
                # Save prototype and images
                proto_file = reg_path + fold[i_fold] + "_prototype.pkl"
                f = open(proto_file, 'wb')
                pickle.dump(HOG, f)
                proto.append(HOG)
                proto_img.append(img_hog_rescale)
                
            if "_2" in fold[i_fold] and "r_2" not in fold[i_fold]:
                IDXX = IDXX + 1
                   

        col_list_palette_pca = sns.xkcd_palette(col_colors1)
        
        #############################################################################################################
        #tSNE_allEmotions variable contains a set of HOGs from a single animal experiencing varius stimuli/emotions.
        pca2 = PCA(n_components=100)
        hogs_all2 = hogs_all[0]
        for i_fold in range(1, len(hogs_all)):
            hogs_all2 = np.vstack([hogs_all2, hogs_all[i_fold]])
        pca2.fit(hogs_all2)
        pcs2 = pca2.fit_transform(hogs_all2)
        tsne2 = TSNE()
        tsne_results2 = tsne2.fit_transform(pcs2)
        g4 = plt.figure()
        plt.scatter(x = tsne_results2[:,0], y = tsne_results2[:,1], color = col_list_palette_pca, alpha = 0.25)
        plt.show()
        g4.savefig(path +  "\\unregister\\tSNE_projections.png")
        
        
        g5 = plt.figure()
        plt.scatter(x = pcs2[:,0], y = pcs2[:,1], color = col_list_palette_pca, alpha = 0.25)
        # plt.xlim(-12,12); plt.ylim(-12,12)
        plt.xlim(-4,4); plt.ylim(-4,4)
        plt.xlabel('PC1'); plt.ylabel('PC2')
        plt.title('Native PC projections')
        g5.savefig(path +  "\\unregister\\PC_projections.png")
        
        X_train = FE.explained_variance(pca2, hogs_all2)
        
        
        loadings = pca2.components_.T * np.sqrt(pca2.explained_variance_)
    
        loading_matrix = pd.DataFrame(loadings)
        PCA_file = path + "\\unregister\\PCA_loadings.pkl"
        f = open(PCA_file, 'wb')
        pickle.dump(loading_matrix, f)
        
        PCA_model_file = path + "\\unregister\\PCA_model.pkl"
        f = open(PCA_model_file, 'wb')
        pickle.dump(pca2, f)  
        
        DIM = img_hog.shape
        DIM = np.floor(np.divide(DIM,pix_cell))
        DIM = DIM.astype(int)
        tot_pix = DIM[0]*DIM[1]
        
        g5 = plt.figure()
        plt.plot(np.cumsum(pca2.explained_variance_ratio_))
        plt.xlabel('# of PCs'); plt.ylabel('Variance Explained (a.u.)')
        plt.ylim(0, 1)
        plt.title('PC Variance Explained')
        g5.savefig(path +  "\\unregister\\PC_Variance Explained.png")
        
        # PCA representations
        for jj in range(0, 2):
            g5 = plt.figure(figsize = (10, 3))
            pc1_tmp = []
            for ii in range(0, 8):
                pc1_tmp.append(loading_matrix[jj])
                pc1_tmp[-1] = pc1_tmp[-1][ii:len(hogs_list[-1][1]):8]
                pc1_tmp[-1] = np.reshape(np.ravel(pc1_tmp[-1]), (int(spout_DIM[0]), int(spout_DIM[1])))
                ax = plt.subplot(2, 4, ii+1)
                ax.imshow(pc1_tmp[-1])
                ax.axis('off')
                ax.set_title("orientation: " + str(ii+1))
                
            g5.savefig(path +  "\\unregister\\PC_" + str(jj+1) + "_orientations.png")
            
            g5 = plt.figure()
            plt.imshow(np.mean(np.absolute(pc1_tmp), axis = 0))
            plt.axis('off')
            plt.title("average")
            g5.savefig(path +  "\\unregister\\PC_" + str(jj+1) + "_average.png")
# %%                   
        ##############################################################################################################
        # Concatenate all baselines and stimuli for global correlation
        global_cluster_file = path + "\\unregister\\Global_cluster.png"
        if not isfile(global_cluster_file):
            colorVec3 = (["lightgray"] * L) * cnt[0] + (["lightgray"] * B) * cnt[0]
            hogs_tmp = hogs_corr[0]
            for i_hog in range(1, len(hogs_corr)):
                hogs_tmp = pd.concat([hogs_tmp, hogs_corr[i_hog]])
                colorVec3 = colorVec3 + ([CC[i_hog]] * L) * cnt[i_hog] + (["lightgray"] * B) * cnt[i_hog]
                
            corr3 = hogs_tmp.T.corr()
            g = sns.clustermap(corr3, robust = True, col_colors = colorVec3, row_colors = colorVec3, rasterized = True)
            g.savefig(global_cluster_file)
       
        ##############################################################################################################            
        # Correlate hogs from all stimuli against each prototype
        scaler = MinMaxScaler()
        
        
        proto[0] = np.average(hogs_list[0][1000:1100], axis = 0)
        
        # proto[0] = hogs_list[0][25]
        # proto[1] = hogs_list[1][9090]
        # proto[2] = hogs_list[2][5430]
        # proto[3] = hogs_list[3][5430]
        
        
        tmp = np.r_[5410, 9070, 12730]
        tmp = np.arange(5410,5400 + 10*3660,3660)
        
        # tmp1 = pd.DataFrame(data = hogs_list[1])
        # proto[1] = np.mean(tmp1.iloc[tmp])
        tmp2 = pd.DataFrame(data = hogs_list[1])
        proto[1] = np.mean(tmp2.iloc[tmp])
        tmp3 = pd.DataFrame(data = hogs_list[2])
        proto[2] = np.mean(tmp3.iloc[tmp])
        
        
        proto_file = path + "\\unregister\\neutral_prototype.pkl"
        f = open(proto_file, 'wb')
        pickle.dump(proto[0], f)
        
        proto_file = path + "\\unregister\\sucrose_prototype.pkl"
        f = open(proto_file, 'wb')
        pickle.dump(proto[1], f)
        
        proto_file = path + "\\unregister\\tail_shock_prototype.pkl"
        f = open(proto_file, 'wb')
        pickle.dump(proto[2], f)
        
        
        
        proto2 = pd.DataFrame(data = proto)
        
        for i_fold in range(1, len(hogs_list)):
        # for i_fold in [2]:
    
            hogs2 = pd.DataFrame(data = hogs_list[i_fold])
            
            g6 = plt.figure(figsize = (25,10))
            
            t1plot = []
            EXP = 25
            # for j_fold in [0,3]:#range(0, len(hogs_list)):
            for j_fold in range(0, len(hogs_list)):
                
                tmp = proto2.iloc[j_fold]
                cut= 0
                t1plot.append(hogs2[cut:].corrwith(tmp, axis = 1))
                t1plot[-1] = t1plot[-1].rolling(window=10).mean()
                t1plot[-1] = np.exp(EXP*t1plot[-1])
                t1plot[-1] = scaler.fit_transform(pd.Series(t1plot[-1]).to_numpy().reshape(-1, 1))     
                
                # t1plot[-1] = (t1plot[-1] - .85)/(.925-.85)
                
                if stim_label[j_fold] == 'quinine':
                    CCC = 'purple'
                elif stim_label[j_fold] == 'salt':
                    CCC = 'orange'
                elif stim_label[j_fold] == 'sucrose':
                    CCC = 'green'
                elif stim_label[j_fold] == 'tail_shock':
                    CCC = 'red'
                elif stim_label[j_fold] == 'neutral':
                    CCC = 'black'
                else:
                    CCC = 'yellow'
                
                plt.figure(g6.number)
                ax = plt.subplot(2, len(hogs_list), j_fold + 1)
                # f, ax = plt.subplots(1, figsize = (25,10))
                plt.plot(t1plot[-1], color = CCC, alpha = 1)
                for i_cnt in range(0, cnt[i_fold]):
                    ax.add_patch(patches.Rectangle((idx[i_fold][i_cnt], 0),L,1,linewidth = 1, facecolor = 'b', alpha = 0.5))
    
                if j_fold == 0:
                    plt.ylabel('Proto Sim (norm. exp ' + str(EXP) + '*delta r)', fontsize = 10)
                    plt.xlabel('time (frames)', fontsize = 10)
                    plt.title('Native Prototype Similarity', fontsize = 15)
                    
                    
                t2plot = []    
                for i_cnt in range(0, cnt[i_fold]):
                    t2plot.append(t1plot[-1][idx[i_fold][i_cnt] - 2*B - cut : idx[i_fold][i_cnt] + 10* L - cut])
                    
                tmp2 = np.mean(t2plot, axis = 0)
                tmp2_std = np.std(t2plot, axis = 0)/np.sqrt(10)
                
                # f, ax = plt.subplots(1, figsize = (25,10))
                ax = plt.subplot(2, len(hogs_list), j_fold + len(hogs_list) + 1)
                ax.add_patch(patches.Rectangle((2*B, 0),60,1,linewidth = 1, facecolor = 'b', alpha = 0.25))
                ax.plot(tmp2, color = CCC, alpha = 1)
                ax.fill_between(np.array(range(0, len(t2plot[0]))), np.squeeze(tmp2 - tmp2_std), np.squeeze(tmp2 + tmp2_std), color = CCC, alpha = 0.5)
                # ax.set_ylabel('Proto Sim')
                # ax.set_xlabel('time (frames)')
                
                if j_fold == 0:
                    plt.ylabel('Proto Sim (norm. exp ' + str(EXP) + '*delta r)', fontsize = 10)
                    plt.xlabel('time (frames)', fontsize = 10)
                    plt.title('Native Prototype Similarity', fontsize = 15)
                    
                g6.savefig(path + "\\unregister\\" + stim_label[i_fold] + "_prototype_vs_all.png")
                
                if j_fold == i_fold:
                    g2 = plt.figure(figsize = (25,5))
                    ax = plt.subplot(1, 3, 1)
                    ax.add_patch(patches.Rectangle((2*B, 0),60,1,linewidth = 1, facecolor = 'b', alpha = 0.25))
                    tmp22 = np.mean(t2plot[0:2], axis = 0)
                    tmp2_std2 = np.std(t2plot[0:2], axis = 0)/np.sqrt(3)
                    ax.plot(tmp22, color = CCC, alpha = 1)
                    ax.fill_between(np.array(range(0, len(t2plot[0]))), np.squeeze(tmp22 - tmp2_std2), np.squeeze(tmp22 + tmp2_std2), color = CCC, alpha = 0.5)
                    plt.ylabel('Proto Sim (norm. exp ' + str(EXP) + '*delta r)', fontsize = 10)
                    plt.xlabel('time (frames)', fontsize = 10)
                    plt.title('Early', fontsize = 15)
                        
                    ax = plt.subplot(1, 3, 2)
                    ax.add_patch(patches.Rectangle((2*B, 0),60,1,linewidth = 1, facecolor = 'b', alpha = 0.25))
                    tmp22 = np.mean(t2plot[3:7], axis = 0)
                    tmp2_std2 = np.std(t2plot[3:7], axis = 0)/np.sqrt(3)
                    ax.plot(tmp22, color = CCC, alpha = 1)
                    ax.fill_between(np.array(range(0, len(t2plot[0]))), np.squeeze(tmp22 - tmp2_std2), np.squeeze(tmp22 + tmp2_std2), color = CCC, alpha = 0.5)
                    plt.title('Middle', fontsize = 15)
                    
                    ax = plt.subplot(1, 3, 3)
                    ax.add_patch(patches.Rectangle((2*B, 0),60,1,linewidth = 1, facecolor = 'b', alpha = 0.25))
                    tmp22 = np.mean(t2plot[8:10], axis = 0)
                    tmp2_std2 = np.std(t2plot[8:10], axis = 0)/np.sqrt(3)
                    ax.plot(tmp22, color = CCC, alpha = 1)
                    ax.fill_between(np.array(range(0, len(t2plot[0]))), np.squeeze(tmp22 - tmp2_std2), np.squeeze(tmp22 + tmp2_std2), color = CCC, alpha = 0.5)
                    plt.title('Late', fontsize = 15)
                    g2.savefig(path + "\\unregister\\" + stim_label[i_fold] + "_temporal_prototypes.png")     
                
         
    
            plt.show()
    
    
        # Classification
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
            
            colorVecClass = 120 * ['gray'] + 600 * ['green'] + 600 * ['red']
            df_col_colors1 = pd.DataFrame(colorVecClass)
            train_labels = df_col_colors1.iloc[list(train_idx)]
            test_labels = df_col_colors1.iloc[list(test_idx)]
            
            train_data = pc_data3.iloc[train_idx]
            test_data = pc_data3.iloc[test_idx]
            
            rf_1 = RandomForestClassifier()
            rf_1.fit(train_data, np.ravel(train_labels))
            predictions_1 = rf_1.predict(test_data)
            
            acc.append(metrics.accuracy_score(test_labels, predictions_1))
            acc_n.append(metrics.accuracy_score(test_labels[0:len(hogs_all[0])], predictions_1[0:len(hogs_all[0])]))
            acc_s.append(metrics.accuracy_score(test_labels[len(hogs_all[0]):len(hogs_all[0])+len(hogs_all[1])], predictions_1[len(hogs_all[0]):len(hogs_all[0])+len(hogs_all[1])]))
            acc_ts.append(metrics.accuracy_score(test_labels[len(hogs_all[1]):len(hogs_all[1])+len(hogs_all[2])], predictions_1[len(hogs_all[1]):len(hogs_all[1])+len(hogs_all[2])]))
            
            
        
        classification_file = path + "\\unregister\\classification.pkl"
        f = open(classification_file, 'wb')
        pickle.dump([acc, acc_n, acc_s, acc_ts], f)
        
        g2 = plt.figure()
        M = [np.mean(acc_n), np.mean(acc_s), np.mean(acc_ts), np.mean(acc)]
        STD = [np.std(acc_n), np.std(acc_s), np.std(acc_ts), np.std(acc)]
        C = ['gray', 'green', 'red', 'black']
        L = ['neutral', 'sucrose', 'tail_shock', 'total']
        
        plt.bar([1, 2, 3, 4], M, yerr = STD, color = C, width = 0.8, tick_label = L)
        plt.ylabel('Classification Accuracy (%)'); plt.ylim(0,1.1)
        plt.title('Native Projection')
        g2.savefig(path + "\\unregister\\Native_Classification.png")
        
        g6 = plt.figure()
        plt.scatter(list(range(0, len(predictions_1))), predictions_1)
        plt. vlines([len(hogs_all[0]), len(hogs_all[0]) + len(hogs_all[1])],-0.25,2.25)
        







