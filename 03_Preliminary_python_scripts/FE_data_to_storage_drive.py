# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:36:58 2020

@author: bedelman
"""


from distutils.dir_util import copy_tree
import os

# path_src = r'C:\Users\Bedelman\Videos\Facial_Exp'
path_src = r'C:\data\experiments\Facial_Exp'
path_dst = r'J:\Bradley Edelman\Facial_Exp_fUS'
path_dst2 = r"H:\Facial_Expression_Videos"

Loc = 1


fold = os.listdir(path_src)
for  i in range(0,len(fold)):
    
    dst = path_dst + "\\" + fold[i]
    dst2 = path_dst2 + "\\" + fold[i]
    src = path_src + "\\" + fold[i]
    
    print(src)
    
    if Loc == 1:
        
        if os.path.isdir(src) and not os.path.exists(dst): # Copy to J: Mace shared drive
            # print(str(i))
            copy_tree(src, dst)
            
    elif Loc == 2: 
        
        if os.path.isdir(src) and not os.path.exists(dst2): # Copy to F: external HDD
            # print(str(i))
            copy_tree(src, dst2)