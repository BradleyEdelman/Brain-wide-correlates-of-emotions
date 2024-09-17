# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import seaborn as sns
import numpy as np  
import matplotlib.pyplot as plt 
from skimage import io
from skimage.feature import hog
from joblib import Parallel, delayed
import cv2
import glob

# %% 
###example image - full frame

import matplotlib.image as mpimg
plt.imshow(mpimg.imread(r"C:\Users\bedelman\Dropbox\Mace_Gogolla_Lab\Facial_Recognition\20200121_mouseFacialExpr_v1\protoFaces_1\neutral\neutral (27).jpg"), cmap="gray")


# %% Importing example data

#folder paths - adjust after saving
neutralFolder = r"C:\Users\Brad\Dropbox\Mace_Gogolla_Lab\Facial_Recognition\20200121_mouseFacialExpr_v1\protoFaces_1\neutral\*.jpg"
painFolder = r"C:\Users\Brad\Dropbox\Mace_Gogolla_Lab\Facial_Recognition\20200121_mouseFacialExpr_v1\protoFaces_1\pain\*.jpg"

suc4pFolder = r"C:\Users\Brad\Dropbox\Mace_Gogolla_Lab\Facial_Recognition\Facial_Expressions\t2_sucrose_4p\*.jpg"
suc20pFolder = r"C:\Users\Brad\Dropbox\Mace_Gogolla_Lab\Facial_Recognition\Facial_Expressions\*t2_sucrose_20p\*.jpg"
shockFolder = r"C:\Users\Brad\Dropbox\Mace_Gogolla_Lab\Facial_Recognition\Facial_Expressions\t2_tailShock\*.jpg"
quinFolder = r"C:\Users\Brad\Dropbox\Mace_Gogolla_Lab\Facial_Recognition\Facial_Expressions\t2_quinine\*.jpg"

#import crop coordinates 
import pickle 
with open(r"C:\Users\Brad\Dropbox\Mace_Gogolla_Lab\Facial_Recognition\20200121_mouseFacialExpr_v1\proto_cropCoords\proto_cropCoords.pkl", 'rb') as f:
     proto_cropCoords = pickle.load(f)    
     

#import prototypical faces
pickleDirPath = r"C:\Users\Brad\Dropbox\Mace_Gogolla_Lab\Facial_Recognition\20200121_mouseFacialExpr_v1\protoFaces/"
from os import listdir
for i in listdir(pickleDirPath):
    vars()[i[:-4]] = pd.read_pickle(pickleDirPath + i)
    
#import tSNE data
pickleDirPath = r"C:\Users\Brad\Dropbox\Mace_Gogolla_Lab\Facial_Recognition\20200121_mouseFacialExpr_v1\tSNE_faces/"
from os import listdir
for i in listdir(pickleDirPath):
    vars()[i[:-4]] = pd.read_pickle(pickleDirPath + i)

# %%example image - cropped

r = proto_cropCoords
plt.imshow(mpimg.imread(r"C:\Users\Brad\Dropbox\Mace_Gogolla_Lab\Facial_Recognition\20200121_mouseFacialExpr_v1\protoFaces_1\neutral\neutral (27).jpg")[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])], cmap="gray")    

# %% Preprocessing

import tmp_func
#replace the filepath below and uncomment for own use. In all following examples a set of preselected coordinated will be used 
#(imported in the cell above as proto_cropCoords)
# coords_1 = tmp_func.findCropCoords(suc4pFolder)


