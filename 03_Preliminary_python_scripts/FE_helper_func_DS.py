# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:51:20 2020

@author: bedelman
"""

# %%

#manually select the area containing the head of the mouse, 
#which will save the coordinates into a variable, to be used later.

def findCropCoords(imgFolderAndFileType):
    ###accepts folder containing image files in format "D:/folder1/folder2/folder3/*.jpg"
    ###waits for user to draw a rectangular selection
    ###outputs coordinates of a rectangular selection drawn over an image
    ###by default, the image displayed is the second image in the input folder
    import cv2
    from skimage import io
    
    coll = io.ImageCollection(imgFolderAndFileType)
    coords1 = cv2.selectROI("Image", coll[0]) 
    
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return coords1

# %%

###Data loading and preprocessing
#the imagesToHogsCellCrop() function will take all image files in a given folder, 
#crop the area as given in the cropCoords (use findCropCoords()) 
#and convert them into their HOG (histogram of oriented gradients) descriptors.
#pixels_per_cell argument defines the sliding window size for HOG creation.
# n_jobs argument defines number of threads used

def imagesToHogsCellCrop(imgFolderAndFileType, pixelsPerCell, cropCoords = []):
    from skimage import io
    from skimage.feature import hog
    from joblib import Parallel, delayed
    
    coll = io.ImageCollection(imgFolderAndFileType)
    
    if cropCoords == []:
        cropCoords = findCropCoords(imgFolderAndFileType)
    
    r = cropCoords
    
    kwargs = dict(orientations = 8, pixels_per_cell = (pixelsPerCell, pixelsPerCell), cells_per_block=(1, 1), transform_sqrt = True)
    return Parallel(n_jobs = 32)(delayed(hog)(image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])], **kwargs) for image in coll)


# %%


### computes variance explained by a PCA model on specified input data
def explained_variance(model, X):
    
    import numpy as np
    from sklearn.decomposition import PCA

    result = np.zeros(model.n_components)
    for ii in range(model.n_components):
        X_trans = model.transform(X)
        X_trans_ii = np.zeros_like(X_trans)
        X_trans_ii[:, ii] = X_trans[:, ii]
        X_approx_ii = model.inverse_transform(X_trans_ii)

        result[ii] = 1 - (np.linalg.norm(X_approx_ii - X) /
                          np.linalg.norm(X - model.mean_)) ** 2
    return result


