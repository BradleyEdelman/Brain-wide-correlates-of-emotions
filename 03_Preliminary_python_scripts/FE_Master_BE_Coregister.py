# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 15:21:15 2020

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
import time
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

template_date = "20201119"
template_id = "mouse_DS_WT_22"
template_mouse = Data + "\\" + template_date + "\\" + template_id

for i_date in range(0, 1): #len(Date)):
    for i_mouse in range(0, len(Mouse[i_date])):

        path = Data + "\\" + Date[i_date] + "\\" + Mouse[i_date][i_mouse]
        reg_path = path + "\\coregister_" + template_date + "_" + template_id + "\\"
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
        proto = []; template_proto = []; coreg_proto = []; coreg_proto_img = []
        hogs_all = []; hogs_list = []; hogs_corr = []; hogs_df = [];
        cnt = []; CC = []; col_colors1 = []
        stim_label = []
        
        pix_cell = 32
        orient = 8
        IDXX = 0 
        for i_fold in range(1, len(fold)):
            
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
                # CROP COORDINATES
                if fold[i_fold] == 'neutral':
                    
                    # TEMPLATE IMAGE
                    template_coord_file = template_mouse + "\\unregister\\crop_coord.pkl"
                    template_spout_img_coord_file = template_mouse + "\\unregister\\spout_img_crop_coord.pkl"    
                    template_spout_hog_coord_file = template_mouse + "\\unregister\\spout_hog_crop_coord.pkl"
                    template_spout_idx_file = template_mouse + "\\unregister\\spout_hog_crop_idx.pkl"
                    template_spout_DIM_file = template_mouse + "\\unregister\\spout_crop_DIM.pkl"
                    if os.path.isfile(template_coord_file):
                        with open(template_coord_file,"rb") as f:
                            template_coord = pickle.load(f)
                            
                            # TEMPLATE SPOUT CROP COORDINATIES 
                            if os.path.isfile(template_spout_img_coord_file) & os.path.isfile(template_spout_hog_coord_file) &  os.path.isfile(template_spout_idx_file) & os.path.isfile(template_spout_DIM_file):
                                with open(template_spout_hog_coord_file,"rb") as ff:
                                    template_spout_hog_coord = pickle.load(ff)
                                
                                with open(template_spout_img_coord_file,"rb") as ff:
                                    template_spout_img_coord = pickle.load(ff)
                                    
                                with open(template_spout_idx_file,"rb") as ff:
                                    template_spout_idx = pickle.load(ff)
                                    
                                with open(template_spout_DIM_file,"rb") as ff:
                                    template_spout_DIM = pickle.load(ff)
                                    
                           
                    else:
                        template_fold = template_mouse + "\\" + fold[i_fold] + "\\FACE\\*.jpg"
                        template_coord = FE.findCropCoords(template_fold)
                        template_spout_coord, template_spout_idx, template_spout_DIM =  FE.findCropCoords2(template_fold, template_coord, pix_cell, orient)
                                                

                    # ORIGINAL IMAGE
                    coord_file = path + "\\unregister\\crop_coord.pkl"
                    spout_img_coord_file = path + "\\unregister\\spout_img_crop_coord.pkl"    
                    spout_hog_coord_file = path + "\\unregister\\spout_hog_crop_coord.pkl"
                    spout_idx_file = path + "\\unregister\\spout_hog_crop_idx.pkl"
                    spout_DIM_file = path + "\\unregister\\spout_crop_DIM.pkl"
                    if os.path.isfile(coord_file):
                        with open(coord_file,"rb") as f:
                            coord = pickle.load(f)
                            
                            # SPOUT CROP COORDINATIES 
                            if os.path.isfile(spout_img_coord_file) & os.path.isfile(spout_hog_coord_file) & os.path.isfile(spout_idx_file) & os.path.isfile(spout_DIM_file):
                                with open(spout_hog_coord_file,"rb") as ff:
                                    spout_hog_coord = pickle.load(ff)
                                    
                                with open(spout_img_coord_file,"rb") as ff:
                                    spout_img_coord = pickle.load(ff)
                                    
                                with open(spout_idx_file,"rb") as ff:
                                    spout_idx = pickle.load(ff)
                                    
                                with open(spout_DIM_file,"rb") as ff:
                                    spout_DIM = pickle.load(ff)
                            
                    else:
                        coord = FE.findCropCoords(path + "\\" + fold[i_fold] + "\\FACE\\*.jpg")
                        coord = FE.findCropCoords(stim_fold)
                        spout_img_coord, spout_hog_coord, spout_idx, spout_DIM =  FE.findCropCoords2(stim_fold, coord, pix_cell, orient)
                         
                    
                    # ESNURE TEMPLATE AND ORIGINAL CROP COORDINATES SAME SIZE
                    template_coord = list(template_coord); template_coord[0] = 0; template_coord[2] = 1280
                    coord = list(coord); coord[0] = 0; coord[2] = 1280;
                    coord[3] = template_coord[3]
                    
                    template_spout_img_coord = list(template_spout_img_coord)
                    if template_spout_img_coord[1] + template_spout_img_coord[3] > template_coord[1] + template_coord[3]:
                        template_spout_img_coord[3] = template_coord[1] + template_coord[3] - template_spout_img_coord[1]

                    spout_img_coord = list(spout_img_coord)
                    if spout_img_coord[1] + spout_img_coord[3] > coord[1] + coord[3]:
                        spout_img_coord[3] = coord[1] + coord[3] - spout_img_coord[1]
                        
                # FIRST COREGISTER NEUTRAL FACE TO TEMPLATE NETURAL FACE, AND THEN EMOTION TO NEUTRAL
                if fold[i_fold] == 'neutral':
                    template_fold = template_mouse +"\\neutral\\FACE\\*.jpg"  
                    # CHECK CROPPED REGISTRATION
                    coreg_param_file = reg_path + "coreg_param.pkl"
                    if os.path.isfile(coreg_param_file):
                         with open(coreg_param_file,"rb") as f:
                             shift1, minErrAngleRot1, shift2, minErrScale1 = pickle.load(f)
                    else:
                        shift1, minErrAngleRot1, shift2, minErrScale1 = FE. findAlignParamsFolder(template_fold, stim_fold, template_coord, coord, template_spout_img_coord, spout_img_coord)
                        f = open(coreg_param_file, 'wb')
                        pickle.dump([shift1, minErrAngleRot1, shift2, minErrScale1], f)
                        
                    image = cv2.imread(glob.glob(template_fold)[1],0); offset_image = cv2.imread(glob.glob(stim_fold)[1],0)
                    image=image[template_coord[1] : template_coord[1] + template_coord[3], template_coord[0] : template_coord[0] + template_coord[2]]
                    offset_image = offset_image[coord[1] : coord[1] + coord[3], coord[0] : coord[0] + coord[2]]
                    
                    scaledImg1 = FE. alignFunc1(offset_image, shift1, minErrAngleRot1, shift2, minErrScale1)
                    
                    a1 = np.dstack((image, offset_image, np.zeros(image.shape).astype("uint8")))
                    a2 = np.dstack((image, scaledImg1, np.zeros(image.shape).astype("uint8")))
                    f, (ax1, ax2) = plt.subplots(ncols = 2, nrows = 1)
                    ax1.imshow(a1); ax1.axis('off'); ax1.set_title("original")
                    ax2.imshow(a2); ax2.axis('off');  ax2.set_title("aligned")
                
            
                
                # EXTRACT HOGS
                hogs_file = reg_path + fold[i_fold] + "_hogs_list.pkl"
                if os.path.isfile(hogs_file):
                    with open(hogs_file,"rb") as f:
                       hogs_load = pickle.load(f)
                    
                    hogs_list.append(hogs_load)
                    
                else:
                
                    # CROPPED REGISTRATION AND HOG EXTRACTION                    
                    coll = io.ImageCollection(stim_fold)
                    num_stack = round(len(coll)/5000)
                    t = time.time()
                    if num_stack == 0:
                        tmp_hogs = FE.imagesToHogsCellCropAlignFolder2(coll, pix_cell, coord, shift1, minErrAngleRot1, shift2, minErrScale1)
                    else:
                        
                        tmp_hogs = []
                        for i_stack in range(0, num_stack):
                            
                            if 5000 + i_stack*5000 < len(coll):
                                stack = coll[0 + i_stack*5000 : 5000 + i_stack*5000]
                            else:
                                stack = coll[0 + i_stack*5000:]
                            
                            tmp_hogs = tmp_hogs + FE.imagesToHogsCellCropAlignFolder2(stack, pix_cell, template_coord, shift1, minErrAngleRot1, shift2, minErrScale1)
                    
                    elapsed = time.time() - t
                    print(fold[i_fold] + " coreg & HoG creation time (sec): " + str(elapsed))
                    hogs_list.append(tmp_hogs)
                    f = open(hogs_file, 'wb')
                    pickle.dump(hogs_list[-1], f)
                        
                    
                    
                    
                # REMOVE SPOUT INDICES FROM HOG
                for i_hog in range(0,len(hogs_list[-1])):
                    hogs_list[-1][i_hog][template_spout_idx] = 0
                
                # CHECK SPOUT REMOVAL FROM HOG
                g5 = plt.figure()
                for ii in range(0, 8):
                    pc1_tmp = hogs_list[-1][2]
                    pc1_tmp = pc1_tmp[ii:len(hogs_list[-1][1]):8]
                    pc1_tmp = np.reshape(pc1_tmp, (int(template_spout_DIM[0]), int(template_spout_DIM[1])))
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
                if not isfile(cluster_file):
                    colorVec1 = ([CC[-1]] * L) * cnt[-1] + (["lightgray"] * B) * cnt[-1] # color vector for cluster map
                    # pairwise correlation between all frames of interest
                    corr = hogs_corr[-1].T.corr()
                    #sns.heatmap(corr, robust = True, rasterized = True)
                    #g = sns.clustermap(corr, robust=True, rasterized = True)
                    g = sns.clustermap(corr, robust=True, col_colors = colorVec1, row_colors = colorVec1, rasterized = True)  
                    g.savefig(reg_path + fold[i_fold] + "_cluster.png")
                    
                    
                
                # TEMPLATE PROTOTYPE INFORMATION
                template_proto_file = template_mouse + "\\unregister\\" + fold[i_fold] + "_prototype.pkl"
                if os.path.isfile(template_proto_file):
                    with open(template_proto_file,"rb") as f:
                       template_proto.append(pickle.load(f))
                
                template_proto_img_fold = template_mouse + "\\" + fold[i_fold] + "_2\\*.jpg"
                template_img = cv2.imread(glob.glob(template_proto_img_fold)[0],0)
                template_img_crop = template_img[template_coord[1]:template_coord[1]+template_coord[3],template_coord[0]:template_coord[0]+template_coord[2]]
                template_HOG, template_img_hog = hog(template_img_crop, orientations = orient, pixels_per_cell = (pix_cell, pix_cell), cells_per_block = (1, 1), visualize = True, transform_sqrt = True)
                template_img_hog_rescale = exposure.rescale_intensity(template_img_hog, in_range=(0, .1))
                
                # ORIGINAL PROTOTYPE INFORMATION
                proto_file = path + "\\unregister\\" + fold[i_fold] + "_prototype.pkl"
                if os.path.isfile(template_proto_file):
                    with open(template_proto_file,"rb") as f:
                       proto.append(pickle.load(f))
                
                proto_img_fold = path + "\\" + fold[i_fold] + "_2\\*.jpg"
                img = cv2.imread(glob.glob(proto_img_fold)[0],0)
                img_crop = img[coord[1]:coord[1]+coord[3],coord[0]:coord[0]+coord[2]]
                HOG, img_hog = hog(img_crop, orientations = orient, pixels_per_cell = (pix_cell, pix_cell), cells_per_block = (1, 1), visualize = True, transform_sqrt = True)
                img_hog_rescale = exposure.rescale_intensity(img_hog, in_range=(0, .1))
                
                # COREGISTERED PROTOTYPE INFORMATION
                coreg_img = FE.alignFunc1(img, shift1, minErrAngleRot1, shift2, minErrScale1)
                coreg_img_crop = FE.alignFunc1(img_crop, shift1, minErrAngleRot1, shift2, minErrScale1)
                coreg_HOG, coreg_img_hog = hog(coreg_img_crop, orientations = orient, pixels_per_cell = (pix_cell, pix_cell), cells_per_block = (1, 1), visualize = True, transform_sqrt = True)
                coreg_img_hog_rescale = exposure.rescale_intensity(coreg_img_hog, in_range=(0, .1))
                coreg_proto_img.append(coreg_img)
                
                if fold[i_fold] == 'neutral':
                    neutral_img_coreg_file = reg_path + fold[i_fold] + "_coreg_image.jpg"
                    neutral_img_crop_coreg_file = reg_path + fold[i_fold] + "_coreg_image_cropped.jpg"
                
                
                coreg_proto_file = reg_path + fold[i_fold] + "_prototype.pkl"
                f = open(coreg_proto_file, 'wb')
                pickle.dump(coreg_HOG, f)
                coreg_proto.append(coreg_HOG)
                
                
                g2 = plt.figure(figsize = (25,20))
                ax1 = plt.subplot(3,3,1); ax1.axis('off')
                ax1.imshow(template_img, cmap = plt.cm.gray);
                ax2 = plt.subplot(3,3,2); ax2.axis('off')
                ax2.imshow(template_img_crop, cmap = plt.cm.gray); ax2.set_title(fold[i_fold] + ' - template')
                ax3 = plt.subplot(3,3,3); ax3.axis('off')
                ax3.imshow(template_img_hog_rescale, cmap = plt.cm.gray);
                
                ax4 = plt.subplot(3,3,4); ax4.axis('off')
                ax4.imshow(img, cmap = plt.cm.gray);
                ax5 = plt.subplot(3,3,5); ax5.axis('off')
                ax5.imshow(img_crop, cmap = plt.cm.gray); ax5.set_title(fold[i_fold] + ' - original')
                ax6 = plt.subplot(3,3,6); ax6.axis('off')
                ax6.imshow(img_hog_rescale, cmap = plt.cm.gray);
                
                ax7 = plt.subplot(3,3,7); ax7.axis('off')
                ax7.imshow(coreg_img, cmap = plt.cm.gray);
                ax8 = plt.subplot(3,3,8); ax8.axis('off')
                ax8.imshow(coreg_img_crop, cmap = plt.cm.gray); ax8.set_title(fold[i_fold] + ' - original coreg')
                ax9 = plt.subplot(3,3,9); ax9.axis('off')
                ax9.imshow(coreg_img_hog_rescale, cmap = plt.cm.gray);
                
                g2.savefig(reg_path + fold[i_fold] + "_HoG.png")
                plt.show()
                
            
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
       
        loadings = pca2.components_.T * np.sqrt(pca2.explained_variance_)
        loading_matrix = pd.DataFrame(loadings)
        load = []
        for i_pc in range(0, loadings.shape[1]):
            # load.append(np.matmul(hogs_all2, loading_matrix[i_pc].to_numpy()))
            load.append(np.matmul(hogs_all2.to_numpy(), loadings.T[i_pc]))
            
        g = plt.figure()
        plt.scatter(load[0], load[1], c = np.asarray(col_list_palette_pca), alpha = 0.5)
        plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Native PC Projections')
        plt.xlim(-12,12); plt.ylim(-12,12)
        g.savefig(reg_path + "Native_PC_projections.png")
        
        template_pca_model_file = template_mouse + "\\unregister\PCA_model.pkl"
        with open(template_pca_model_file,"rb") as f:
            template_pca = pickle.load(f)
            
        X_test = FE.explained_variance(template_pca, hogs_all2)
        g5 = plt.figure()
        plt.plot(np.cumsum(X_test))
        plt.xlabel('# of PCs'); plt.ylabel('Variance Explained (a.u.)')
        plt.ylim(0, 1)
        plt.title('PC Variance Explained')
        g5.savefig(reg_path + "Template_PC_Variance_Explained.png")
        
        template_pca_file = template_mouse + "\\unregister\PCA_loadings.pkl"
        with open(template_pca_file,"rb") as f:
            template_loadings = pickle.load(f)
            
        template_loading_matrix = pd.DataFrame(template_loadings)
        template_load = []
        for i_pc in range(0, template_loadings.shape[1]):
            template_load.append(np.matmul(hogs_all2, template_loading_matrix[i_pc].to_numpy()))
            
        g = plt.figure()
        plt.scatter(template_load[0], template_load[1], c = np.asarray(col_list_palette_pca), alpha = 0.5)
        plt.xlabel('PC1'); plt.ylabel('PC2'); plt.title('Template PC Projections')
        # plt.xlim(-12,12); plt.ylim(-12,12)
        plt.xlim(-4,4); plt.ylim(-4,4)
        g.savefig(reg_path + "Template_PC_projections.png")
        
        # Template PCA
        for jj in range(0, 2):
            g5 = plt.figure(figsize = (10, 3))
            pc1_tmp = []
            for ii in range(0, 8):
                pc1_tmp.append(template_loadings[jj])
                pc1_tmp[-1] = pc1_tmp[-1][ii:len(hogs_list[-1][1]):8]
                pc1_tmp[-1] = np.reshape(np.ravel(pc1_tmp[-1]), (int(template_spout_DIM[0]), int(template_spout_DIM[1])))
                ax = plt.subplot(2, 4, ii+1)
                ax.imshow(pc1_tmp[-1])
                ax.axis('off')
                ax.set_title("orientation: " + str(ii+1))
            
            g5 = plt.figure()
            plt.imshow(np.mean(np.absolute(pc1_tmp), axis = 0))
            plt.axis('off')
            plt.title("average")
  
        ##############################################################################################################            
        # Correlate hogs from all stimuli against each prototype
        scaler = MinMaxScaler()
        
# %%
        proto[0] = np.average(hogs_list[0][1000:1100], axis = 0)
        
        tmp = np.arange(5410,5400 + 10*3660,3660)
        
        tmp2 = pd.DataFrame(data = hogs_list[1])
        proto[1] = np.mean(tmp2.iloc[tmp])
        tmp3 = pd.DataFrame(data = hogs_list[2])
        proto[2] = np.mean(tmp3.iloc[tmp])
        
        
        proto_file = template_mouse + "\\unregister\\neutral_prototype.pkl"
        with open(proto_file,"rb") as f:
            template_proto[0] = pickle.load(f)
        
        proto_file = template_mouse + "\\unregister\\sucrose_prototype.pkl"
        with open(proto_file,"rb") as f:
            template_proto[1] = pickle.load(f)
            
        proto_file = template_mouse + "\\unregister\\tail_shock_prototype.pkl"
        with open(proto_file,"rb") as f:
            template_proto[2] = pickle.load(f)
        
        proto2 = pd.DataFrame(data = proto)
        template_proto2 = pd.DataFrame(data = template_proto)
        
        for i_fold in range(1, len(hogs_list)):
        # for i_fold in [2]:
    
            hogs2 = pd.DataFrame(data = hogs_list[i_fold])
            # hogs2 = hogs2.iloc[1801:-1800]
            
            g6 = plt.figure(figsize = (25,10))
            
            t1plot = []
            template_t1plot = []
            
            EXP = 25
            # for j_fold in [0,3]:#range(0, len(hogs_list)):
            for j_fold in range(0, len(hogs_list)):
                
                tmp = proto2.iloc[j_fold]
                t1plot.append(hogs2.corrwith(tmp, axis = 1))
                t1plot[-1] = t1plot[-1].rolling(window=10).mean()
                t1plot[-1] = np.exp(EXP*t1plot[-1])
                t1plot[-1] = scaler.fit_transform(pd.Series(t1plot[-1]).to_numpy().reshape(-1, 1)) 
                
                template_tmp = template_proto2.iloc[j_fold]
                template_t1plot.append(hogs2.corrwith(template_tmp, axis = 1))
                template_t1plot[-1] = template_t1plot[-1].rolling(window=10).mean()
                template_t1plot[-1] = np.exp(EXP*template_t1plot[-1])
                template_t1plot[-1] = scaler.fit_transform(pd.Series(template_t1plot[-1]).to_numpy().reshape(-1, 1)) 
                
                if stim_label[j_fold] == 'quinine':
                    CCC = 'purple'; CCC1 = 'indigo'
                elif stim_label[j_fold] == 'salt':
                    CCC = 'orange';
                elif stim_label[j_fold] == 'sucrose':
                    CCC = 'green'; CCC1 = 'darkgreen'
                elif stim_label[j_fold] == 'tail_shock':
                    CCC = 'red'; CCC1 = 'darkred'
                elif stim_label[j_fold] == 'neutral':
                    CCC = 'gray'; CCC1 = 'black'
                else:
                    CCC = 'yellow'
                
                # ax = plt.subplot(2, len(hogs_list), j_fold + 1)
                # for i_cnt in range(0, cnt[i_fold]):
                #     ax.add_patch(patches.Rectangle((idx[i_fold][i_cnt], 0),60,1,linewidth = 1, facecolor = 'b', alpha = 0.5))
                # ax.plot(t1plot[-1], color = CCC, alpha = 1)
                    
                t2plot = []    
                for i_cnt in range(0, cnt[i_fold]):
                    t2plot.append(t1plot[-1][idx[i_fold][i_cnt] - 2*B : idx[i_fold][i_cnt] + 10* L])
                    
                tmp2 = np.mean(t2plot, axis = 0)
                tmp2_std = np.std(t2plot/np.sqrt(10), axis = 0)
                
                template_t2plot = []    
                for i_cnt in range(0, cnt[i_fold]):
                    template_t2plot.append(template_t1plot[-1][idx[i_fold][i_cnt] - 2*B : idx[i_fold][i_cnt] + 10* L])
                    
                template_tmp2 = np.mean(template_t2plot, axis = 0)
                template_tmp2_std = np.std(template_t2plot/np.sqrt(10), axis = 0)
                
                ax = plt.subplot(2, len(hogs_list), j_fold + 1)
                ax.add_patch(patches.Rectangle((0, 0),60,1,linewidth = 1, facecolor = 'b', alpha = 0.25))
                ax.plot(list(range(-2*B, 10*L)), tmp2, color = CCC, alpha = 1)
                ax.fill_between(np.array(range(-2*B, 10*L)), np.squeeze(tmp2 - tmp2_std), np.squeeze(tmp2 + tmp2_std), color = CCC, alpha = 0.5)
                # ax.plot(list(range(-2*B, 10*L)), template_tmp2, color = CCC1, alpha = 1)
                # ax.fill_between(np.array(range(-2*B, 10*L)), np.squeeze(template_tmp2 - template_tmp2_std), np.squeeze(template_tmp2 + template_tmp2_std), color = CCC1, alpha = 0.5)
                
                if j_fold == 0:
                    plt.ylabel('Proto Sim (norm. exp ' + str(EXP) + '*delta r)', fontsize = 10)
                    plt.xlabel('time (frames)', fontsize = 10)
                    plt.title('Prototype Similarity', fontsize = 15)
                
                ax = plt.subplot(2, len(hogs_list), j_fold + len(hogs_list) + 1)
                ax.add_patch(patches.Rectangle((0, 0),60,1,linewidth = 1, facecolor = 'b', alpha = 0.25))
                ax.plot(list(range(-2*B, 10*L)), template_tmp2, color = CCC, alpha = 1)
                ax.fill_between(np.array(range(-2*B, 10*L)), np.squeeze(template_tmp2 - template_tmp2_std), np.squeeze(template_tmp2 + template_tmp2_std), color = CCC, alpha = 0.5)
                if j_fold == 0:
                    plt.ylabel('Proto Sim (norm. exp ' + str(EXP) + '*delta r)', fontsize = 10)
                    plt.xlabel('time (frames)', fontsize = 10)
                    plt.title('Template Prototype Similarity', fontsize = 15)
                
                g6.savefig(reg_path + stim_label[i_fold] + "_prototype_vs_all.png")
                plt.show()
    
        
        # Classification
        all_data = pd.concat([hogs_all[0], hogs_all[1], hogs_all[2]])
        
        # project hog data onto first two principle components
        pc_data1 = pd.DataFrame(np.matmul(all_data.to_numpy(),loading_matrix[0].to_numpy()))
        pc_data2 = pd.DataFrame(np.matmul(all_data.to_numpy(),loading_matrix[1].to_numpy()))
        pc_data3 = pd.concat([pc_data1, pc_data2], axis = 1)
        
        template_PCA_loading_file = template_mouse + "\\unregister\\PCA_loadings.pkl"
        with open(template_PCA_loading_file,"rb") as f:
            template_loading_matrix = pickle.load(f)
            
        template_pc_data1 = pd.DataFrame(np.matmul(all_data.to_numpy(),template_loading_matrix[0].to_numpy()))
        template_pc_data2 = pd.DataFrame(np.matmul(all_data.to_numpy(),template_loading_matrix[1].to_numpy()))
        template_pc_data3 = pd.concat([template_pc_data1, template_pc_data2], axis = 1)
        
        
        # 100-fold cross validation
        acc = []; acc_n = []; acc_s =[]; acc_ts = []
        template_acc = []; template_acc_n = []; template_acc_s =[]; template_acc_ts = []
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
            
            # coregistered native space classifier
            train_data = pc_data3.iloc[train_idx]
            test_data = pc_data3.iloc[test_idx]
            rf_1 = RandomForestClassifier()
            rf_1.fit(train_data, np.ravel(train_labels))
            
            # coregistered template space classifier
            template_train_data = template_pc_data3.iloc[train_idx]
            template_test_data= template_pc_data3.iloc[test_idx]
            template_rf_1 = RandomForestClassifier()
            template_rf_1.fit(template_train_data, np.ravel(train_labels))
            
            predictions_1 = rf_1.predict(test_data)
            template_predictions_1 = template_rf_1.predict(template_test_data)
            
            acc.append(metrics.accuracy_score(test_labels, predictions_1))
            acc_n.append(metrics.accuracy_score(test_labels[0:len(hogs_all[0])], predictions_1[0:len(hogs_all[0])]))
            acc_s.append(metrics.accuracy_score(test_labels[len(hogs_all[0]):len(hogs_all[0])+len(hogs_all[1])], predictions_1[len(hogs_all[0]):len(hogs_all[0])+len(hogs_all[1])]))
            acc_ts.append(metrics.accuracy_score(test_labels[len(hogs_all[1]):len(hogs_all[1])+len(hogs_all[2])], predictions_1[len(hogs_all[1]):len(hogs_all[1])+len(hogs_all[2])]))
            
            template_acc.append(metrics.accuracy_score(test_labels, template_predictions_1))
            template_acc_n.append(metrics.accuracy_score(test_labels[0:len(hogs_all[0])], template_predictions_1[0:len(hogs_all[0])]))
            template_acc_s.append(metrics.accuracy_score(test_labels[len(hogs_all[0]):len(hogs_all[0])+len(hogs_all[1])], template_predictions_1[len(hogs_all[0]):len(hogs_all[0])+len(hogs_all[1])]))
            template_acc_ts.append(metrics.accuracy_score(test_labels[len(hogs_all[1]):len(hogs_all[1])+len(hogs_all[2])], template_predictions_1[len(hogs_all[1]):len(hogs_all[1])+len(hogs_all[2])]))
        
            
            
        # plt.scatter(list(range(0, len(predictions_1))), predictions_1)
        # plt.vlines([len(hogs_all[0]), len(hogs_all[0]) + len(hogs_all[1])],-0.25,2.25)
        
        # plt.scatter(list(range(0, len(template_predictions_1))), template_predictions_1)
        # plt.vlines([len(hogs_all[0]), len(hogs_all[0]) + len(hogs_all[1])],-0.25,2.25)
            
        g2 = plt.figure()
        M = [np.mean(acc_n), np.mean(acc_s), np.mean(acc_ts), np.mean(acc)]
        STD = [np.std(acc_n), np.std(acc_s), np.std(acc_ts), np.std(acc)]
        C = ['gray', 'green', 'red', 'black']
        L = ['neutral', 'sucrose', 'tail_shock', 'total']
        
        plt.bar([1, 2, 3, 4], M, yerr = STD, color = C, width = 0.8, tick_label = L)
        plt.ylabel('Classification Accuracy (%)'); plt.ylim(0,1.1)
        plt.title('Native Projection')
        g2.savefig(reg_path + "Native_Classification.png")
        
        
        g3 = plt.figure()
        M = [np.mean(template_acc_n), np.mean(template_acc_s), np.mean(template_acc_ts), np.mean(template_acc)]
        STD = [np.std(template_acc_n), np.std(template_acc_s), np.std(acc_ts), np.std(template_acc)]
        C = ['gray', 'green', 'red', 'black']
        L = ['neutral', 'sucrose', 'tail_shock', 'total']
        
        plt.bar([1, 2, 3, 4], M, yerr = STD, color = C, width = 0.8, tick_label = L)
        plt.ylabel('Classification Accuracy (%)'); plt.ylim(0,1.1)
        plt.title('Template Projection')
        g3.savefig(reg_path + "Template_Classification.png")
        
        
        
        classification_file = reg_path + "classification.pkl"
        f = open(classification_file, 'wb')
        pickle.dump([acc, acc_n, acc_s, acc_ts], f)
        
        template_classification_file = reg_path + "template_classification.pkl"
        f = open(template_classification_file, 'wb')
        pickle.dump([template_acc, template_acc_n, template_acc_s, template_acc_ts], f)
        
    
    
    
    
    
    
    
    
    
