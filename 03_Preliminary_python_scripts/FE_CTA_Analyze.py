# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:36:32 2021

@author: bedelman
"""

import pandas as pd
import seaborn as sns
import numpy as np  
import os
from os.path import isfile, isdir, join
import pickle
import FE_helper as FE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# %%

# Specify data path details
multistim = 0 # 1 for yes, 0 for no
Data = r"J:\Bradley Edelman\Facial_Exp_fUS"

Date = ["20210510", "20210511", "20210512", "20210513", "20210514", "20210515"]
Mouse = ["mouse_0119", "mouse_0120"]

Date = ["20210411", "20210412", "20210413", "20210414", "20210415", "20210416"]
Mouse = ["mouse_0070", "mouse_0071", "mouse_0072", "mouse_0073"]


rewrite = 1
register = 1

[IDX_stim, IDX_base, IDX_extend] = FE.idxExtract(3600, 2440, 10, 40, 80)       

pix_cell = 24
for i_mouse in range(1, len(Mouse)):
    
    fold =[]; stim = [];
    hogs_all = []; hogs_stim = []; hogs_cluster = []; hogs_disp = [];
    
    # find all folders for this mouse, across specified days
    for i_date in range(0, len(Date)):
        
        path = Data + "\\" + Date[i_date] + "\\" + Mouse[i_mouse]
        stim.append([x for x in os.listdir(path) if isdir(join(path, x))])
        fold.append(path + "\\" + stim[-1][0] + "\\FACE")
        
    # %%
    # Specify fold idx for building pleasure/disgust protoypes
    spout_info = []
    spout_file = fold[0] + "\\spout_crop_coord.pkl"
    if os.path.isfile(spout_file):
        with open(spout_file, "rb") as f:
            spout_info = pickle.load(f)
    
    proto = []; proto_idx = [0, 2]; # [disgust pleasure]
    for i_idx in range(0, len(proto_idx)):
        
        # Load hogs and remove spout indices
        hog_file = fold[proto_idx[i_idx]] + "\\hogs_list.pkl"
        hog_nospout = FE.hogLoadAndSpoutRemove(hog_file, 8, spout_info)
        hog_nospout = pd.DataFrame(data = hog_nospout)
        
        # Define neutral hog from before mouse received any stimulus
        if i_idx == 0:
            proto_neutral = np.mean(hog_nospout.iloc[np.r_[2200:2300]])

        proto_hog = hog_nospout.iloc[np.r_[IDX_stim]]
        proto_corr = proto_hog.corrwith(proto_neutral, axis = 1)
        proto_corr = proto_corr.rolling(window=10).mean()
        proto_corridx = proto_corr.nsmallest(10).index
        proto_tmp = np.mean(hog_nospout.iloc[np.r_[proto_corridx]])
        
        PLT = FE.hogOrientationViz(proto_tmp.to_numpy(), 8, spout_info)
        PLT.suptitle("proto_" + str(i_idx), fontsize = 35)
        
        proto.append(proto_tmp) 
    
    hog_nospout = []
    for i_fold in range(0, len(fold)):
        
        # Load hogs and remove spout indices
        hog_file = fold[i_fold] + "\\hogs_list.pkl"
        hog_nospout = (FE.hogLoadAndSpoutRemove(hog_file, 8, spout_info))
        hog_nospout = pd.DataFrame(data = hog_nospout)
        
        hogs_all = hog_nospout
        hogs_stim = hog_nospout.iloc[np.r_[IDX_stim]]
        hogs_cluster = hog_nospout.iloc[np.r_[IDX_stim + IDX_base]]
        hogs_disp = hog_nospout.iloc[np.r_[IDX_extend]]
        
        # cluster baseline and stimulation hogs
        cluster_file = fold[i_fold] + "\\cluster.png"
        if not isfile(cluster_file) or rewrite == 1:
            labels = len(IDX_stim)*["blue"] + len(IDX_base)*["lightgray"]
            F = np.int(len(IDX_stim)/4)
            labels = F*['plum'] + F*['mediumorchid'] + F*['blueviolet'] + F*['purple'] + len(IDX_base)*["lightgray"]  
            labels = F*['lightgreen'] + F*['lime'] + F*['seagreen'] + F*['darkgreen'] + len(IDX_base)*["lightgray"]  
            
            hogs_corr = hogs_cluster.T.corr()
            g = sns.clustermap(hogs_corr, robust=True, col_colors = labels, row_colors = labels, rasterized = True)  
            g.savefig(cluster_file)
        
        
        f1 = plt.figure(i_fold + 1)
        f2 = plt.figure(i_fold + 1 + len(fold))
        
        tmp2 = []; tmp2_std = [];
        CC = ['gray', 'purple', 'green']
        for i_proto in range(0, 3):
            if i_proto == 0:
                proto_corr = hogs_all.corrwith(proto_neutral, axis = 1)
            else:
                proto_corr = hogs_all.corrwith(proto[i_proto - 1], axis = 1)
            
            proto_corr = pd.DataFrame(data = proto_corr)
            proto_corr = proto_corr.rolling(window = 10).mean()
            proto_corr = proto_corr.to_numpy()
            proto_corr = scaler.fit_transform(proto_corr.reshape(-1, 1))           
            proto_disp = proto_corr[IDX_extend]
            proto_disp = proto_disp.reshape(10, np.int(len(proto_disp)/10))
            
            tmp = np.mean(proto_disp, axis = 0)
            tmp_std = np.std(proto_disp/np.sqrt(10), axis = 0)
            
            plt.figure(1)
            plt.subplot(1, 3, i_proto + 1)
            plt.plot(tmp, color = CC[i_proto], alpha = 1)
            plt.fill_between(np.array(range(0, len(tmp))), np.squeeze(tmp - tmp_std), np.squeeze(tmp + tmp_std), color = CC[i_proto], alpha = 0.5)
            plt.ylim(0, 1)
            
            proto_stim = proto_corr[IDX_stim]
            tmp2.append(np.mean(proto_stim, axis = 0))
            tmp2_std.append(np.std(proto_stim/np.sqrt(10), axis = 0))
        
        f1.savefig(fold[i_fold] + "\\Prototype_corr.png")
        
        plt.figure(2)
        C = ['gray', 'purple', 'green']
        L = ['neutral', 'quinine', 'sucrose']
        plt.bar([1, 2, 3], np.concatenate(tmp2), yerr = np.concatenate(tmp2_std), color = C, width = 0.8, tick_label = L)
        plt.ylim(0, 1)
        f2.savefig(fold[i_fold] + "\\Prototype_stim_corr.png")
        
        
        