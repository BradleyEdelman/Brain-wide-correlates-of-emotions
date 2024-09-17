# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:00:15 2021

@author: bedelman
"""

import os
from os.path import isfile, isdir, join
import shutil

loc_orig = r'C:\data\experiments\Facial_Exp'
loc_dest = r'C:\data\experiments\Facial_Exp_Movies'

Date = ["20210510", "20210511", "20210512", "20210513", "20210514", "20210515"]
Mouse = ["mouse_0119", "mouse_0120"]
Study = ["CTA"]

Date = ["20210411", "20210412", "20210413", "20210414", "20210415", "20210416"]
Mouse = ["mouse_0070", "mouse_0071", "mouse_0072", "mouse_0073"]
Study = ["CUP"]

Date = ["20210523", "20210524", "20210525", "20210526", "20210527", "20210528"]
Mouse = ["mouse_0122", "mouse_0124", "mouse_0125", "mouse_0127", "mouse_0130"]
Study = ["3Pin"]

Date = ["20210531", "20210601", "20210602", "20210603", "20210607", "20210608"]
Mouse = ["mouse_0137", "mouse_0140", "mouse_0142"]
Study = ["CTA2"]

Date = ["20210621", "20210622", "20210623", "20210624", "20210628", "20210629"]
Mouse = ["mouse_0167", "mouse_0169", "mouse_0170", "mouse_0171"]
Study = ["CTA_3Pin1"]


Date = ["20210818", "20210819"]
Mouse = ["mouse_191", "mouse_192", "mouse_193", "mouse_196", "mouse_198", "mouse_199"]
Study = ["FC"]

Date = ["20210930", "20211001", "20211005"]
Mouse = ["mouse_191", "mouse_192", "mouse_198", "mouse_199"]
Study = ["Odor_test"]


loc_dest = loc_dest + '\\' + Study[0] + '\\'
if not os.path.isdir(loc_dest):
    os.mkdir(loc_dest, 0o777)

Type = ['BODY', 'FACE', 'PUPIL']
for i_type in range(0, len(Type)):
    type_dest = loc_dest  + Type[i_type] + '\\'
    if not os.path.isdir(type_dest):
        os.mkdir(type_dest, 0o777)

for i_date in range(0, len(Date)):
    for i_mouse in range(0, len(Mouse)):
        
        mouse_fold = loc_orig + '\\' + Date[i_date] + '\\' + Mouse[i_mouse] + '\\'
        if os.path.isdir(mouse_fold):
            # stim = os.listdir(mouse_fold)
            stim = [x for x in os.listdir(mouse_fold) if isdir(join(mouse_fold, x))]
            for i_stim in range(0, len(stim)):
                
                stim_fold = mouse_fold + stim[i_stim] + '\\'
                for i_type in range(0, len(Type)):
                    
                    vid_fold = stim_fold + Type[i_type] + '\\'
                    vid_orig_name = [x for x in os.listdir(vid_fold) if isfile(join(vid_fold, x)) and ".avi" in x]
                    for i_vid in range(0, len(vid_orig_name)):
                    
                        vid_orig = stim_fold + Type[i_type] + '\\' + vid_orig_name[i_vid]
                        vid_id = Mouse[i_mouse] + '_' + Date[i_date] + '_' + stim[i_stim] + '_' + vid_orig_name[i_vid]
                        vid_dest = loc_dest + Type[i_type] + '\\' + vid_id
                        if os.path.isfile(vid_orig) and not os.path.isfile(vid_dest):
                            shutil.copyfile(vid_orig, vid_dest)
                            print(vid_dest)