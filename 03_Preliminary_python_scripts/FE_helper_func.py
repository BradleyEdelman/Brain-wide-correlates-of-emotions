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
def findCropCoords2(imgFolderAndFileType, cropCoords, pix_cell, orient):
    ###accepts folder containing image files in format "D:/folder1/folder2/folder3/*.jpg"
    ###waits for user to draw a rectangular selection
    ###outputs coordinates of a rectangular selection drawn over an image
    ###by default, the image displayed is the second image in the input folder
    import cv2
    from skimage import io
    import numpy as np 
    
    coll = io.ImageCollection(imgFolderAndFileType)
    
    tmp = coll[50][cropCoords[1]:cropCoords[1]+cropCoords[3],cropCoords[0]:cropCoords[0]+cropCoords[2]]
    
    coords2 = cv2.selectROI("Image", tmp)
    
    DIM = tmp.shape
    DIM = np.floor(np.divide(DIM, pix_cell))
    DIM = DIM.astype(int)
    
    coords3 = [];
    for i in range(0 ,4):
        coords3.append(np.ceil(int(coords2[i])/pix_cell))
        
    # determine indices of HOG to remove based on these coordinates
    
                
    tmp2 = np.ones(int(DIM[0])*int(DIM[1])*orient)
                
    for ii in range(0, orient):
        tmp_idx = tmp2[ii:len(tmp2):orient]
        tmp_idx = np.reshape(tmp_idx, (int(DIM[0]), int(DIM[1])))
        tmp_idx[int(coords3[1]):int(coords3[1])+int(coords3[3]),int(coords3[0]):int(coords3[0])+int(coords3[2])] = 0
        tmp_idx = np.reshape(tmp_idx, (1,int(DIM[0]) * int(DIM[1])))
        tmp2[ii:len(tmp2):orient] = tmp_idx
        
    idxz = np.where(tmp2 == 0)[0]
    
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return coords2, coords3, idxz, DIM

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

### this function will attempt to find optimal alignment parameters for a pair of images by employing 
### cross-correlation measurement in the fourier space
def findAlignParams(image, offset_image, image_cropCoords, offset_image_cropCoords, image_spout_cropCoords, offset_image_spout_cropCoords):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import cv2

    from skimage.feature import register_translation
    from scipy.ndimage import fourier_shift


    c1 = image_cropCoords
    c11 = image_spout_cropCoords
    if c1 != []:
         image = image[c1[1] : c1[1] + c1[3], c1[0] : c1[0] + c1[2]]
         image[c11[1] : c11[1] + c11[3], c11[0] : c11[0] + c11[2]] = 0
    
    c2 = offset_image_cropCoords
    c22 = offset_image_spout_cropCoords
    if c2 != []:
         offset_image = offset_image[c2[1] : c2[1] + c2[3], c2[0] : c2[0] + c2[2]]
         offset_image[c22[1] : c22[1] + c22[3], c22[0] : c22[0] + c22[2]] = 0

    # first rought XY alignment
    shift1, error1, diffphase1 = register_translation(image, offset_image)
    
    # apply correction
    offset_image2 = fourier_shift(np.fft.fftn(offset_image), shift1)
    offset_image2 = np.fft.ifftn(offset_image2)
    offset_image2 = np.array(offset_image2.real)
    offset_image2 = offset_image2.astype("uint8")
    
    # get image height, width
    (h, w) = image.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    scale = 1.0
    
    # find rotational alignment
    rotAngle1 = []
    rotError1 = []
    for i in range(-45,45):
        M = cv2.getRotationMatrix2D(center, i, 1)
        rotated_i = cv2.warpAffine(offset_image2, M, (w, h))
        
        shiftRot1, errorRot1, diffphaseRot1 = register_translation(image, rotated_i)
        
        rotAngle1.append(i)
        rotError1.append(errorRot1)
    
    # detect angle with the lowest error
    rot1_df = [rotAngle1, rotError1]
    rot1_df = pd.DataFrame(rot1_df)
    rot1_df = rot1_df.T
    minErrRowRot1 = rot1_df[1].idxmin()
    minErrAngleRot1 = rot1_df[0][minErrRowRot1]
    
    # apply rotational correction
    M = cv2.getRotationMatrix2D(center, minErrAngleRot1, 1)
    rotated1 = cv2.warpAffine(offset_image2, M, (w, h))
    
    # check for XY alignment again
    shift2, error2, diffphase2 = register_translation(image, rotated1)
    
    # apply XY correction again
    offset_image3 = fourier_shift(np.fft.fftn(rotated1), shift2)
    offset_image3 = np.fft.ifftn(offset_image3)
    offset_image3 = np.array(offset_image3.real)
    offset_image3 = offset_image3.astype("uint8")
    
    # check for scale alignment
    scale1 = []
    scaleError1 = []
    for i in np.arange(0.0, 2.0, 0.01):
        M = cv2.getRotationMatrix2D(center, 0, i)
        rescaled_i = cv2.warpAffine(offset_image3, M, (w, h))
        
        shiftScale1, errorScale1, diffphaseScale1 = register_translation(image, rescaled_i)
        
        scale1.append(i)
        scaleError1.append(errorScale1)
        
    scale1_df = [scale1, scaleError1]
    scale1_df = pd.DataFrame(scale1_df)
    scale1_df = scale1_df.T
    minErrRowScale1 = scale1_df[1].idxmin()
    minErrScale1 = scale1_df[0][minErrRowScale1]
    
    M = cv2.getRotationMatrix2D(center, 0, minErrScale1)
    scaledImg1 = cv2.warpAffine(offset_image3, M, (w, h))    
    
    
    return shift1, minErrAngleRot1, shift2, minErrScale1


# %%

### this function will attempt to find optimal alignment parameters for a pair of FOLDERS by employing 
### cross-correlation measurement in the fourier space
def findAlignParamsFolder(imageFolder, offset_imageFolder, image_cropCoords, offset_image_cropCoords, image_spout_cropCoords, offset_image_spout_cropCoords):
    
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import cv2
    import glob

    from skimage.feature import register_translation
    from scipy.ndimage import fourier_shift
    
    image = cv2.imread(glob.glob(imageFolder)[1],0)
    c1 = image_cropCoords
    c11 = image_spout_cropCoords
    if c1 != []:
         image = image[c1[1] : c1[1] + c1[3], c1[0] : c1[0] + c1[2]]
         image[c11[1] : c11[1] + c11[3], c11[0] : c11[0] + c11[2]] = 0
    
    offset_image = cv2.imread(glob.glob(offset_imageFolder)[1],0)
    c2 = offset_image_cropCoords
    c22 = offset_image_spout_cropCoords
    if c2 != []:
         offset_image = offset_image[c2[1] : c2[1] + c2[3], c2[0] : c2[0] + c2[2]]
         offset_image[c22[1] : c22[1] + c22[3], c22[0] : c22[0] + c22[2]] = 0
    
    # first rought XY alignment
    shift1, error1, diffphase1 = register_translation(image, offset_image)
    
    # apply correction
    offset_image2 = fourier_shift(np.fft.fftn(offset_image), shift1)
    offset_image2 = np.fft.ifftn(offset_image2)
    offset_image2 = np.array(offset_image2.real)
    offset_image2 = offset_image2.astype("uint8")
    
    # get image height, width
    (h, w) = image.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    scale = 1.0
    
    # find rotational alignment
    rotAngle1 = []
    rotError1 = []
    for i in range(-45,45):
        M = cv2.getRotationMatrix2D(center, i, 1)
        rotated_i = cv2.warpAffine(offset_image2, M, (w, h))
        
        shiftRot1, errorRot1, diffphaseRot1 = register_translation(image, rotated_i)
        
        rotAngle1.append(i)
        rotError1.append(errorRot1)
    
    # detect angle with the lowest error
    rot1_df = [rotAngle1, rotError1]
    rot1_df = pd.DataFrame(rot1_df)
    rot1_df = rot1_df.T
    minErrRowRot1 = rot1_df[1].idxmin()
    minErrAngleRot1 = rot1_df[0][minErrRowRot1]
    
    # apply rotational correction
    M = cv2.getRotationMatrix2D(center, minErrAngleRot1, 1)
    rotated1 = cv2.warpAffine(offset_image2, M, (w, h))
    
    # check for XY alignment again
    shift2, error2, diffphase2 = register_translation(image, rotated1)
    
    # apply XY correction again
    offset_image3 = fourier_shift(np.fft.fftn(rotated1), shift2)
    offset_image3 = np.fft.ifftn(offset_image3)
    offset_image3 = np.array(offset_image3.real)
    offset_image3 = offset_image3.astype("uint8")
    
    # check for scale alignment
    scale1 = []
    scaleError1 = []
    for i in np.arange(0.0, 2.0, 0.01):
        M = cv2.getRotationMatrix2D(center, 0, i)
        rescaled_i = cv2.warpAffine(offset_image3, M, (w, h))
        
        shiftScale1, errorScale1, diffphaseScale1 = register_translation(image, rescaled_i)
        
        scale1.append(i)
        scaleError1.append(errorScale1)
        
    scale1_df = [scale1, scaleError1]
    scale1_df = pd.DataFrame(scale1_df)
    scale1_df = scale1_df.T
    minErrRowScale1 = scale1_df[1].idxmin()
    minErrScale1 = scale1_df[0][minErrRowScale1]
    
    M = cv2.getRotationMatrix2D(center, 0, minErrScale1)
    scaledImg1 = cv2.warpAffine(offset_image3, M, (w, h))    
    
    
    return shift1, minErrAngleRot1, shift2, minErrScale1
    

# %%

###This is a utility function which takes the parameters output by findAlignParams() function
###and uses them to transform and align a given unaligned image
def alignFunc1(offset_image, shift1, minErrAngleRot1, shift2, minErrScale1):
    
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.feature import register_translation
    from scipy.ndimage import fourier_shift
    import cv2
    
    # get image height, width
    (h, w) = offset_image.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    scale = 1.0
    
    offset_image2 = fourier_shift(np.fft.fftn(offset_image), shift1)
    offset_image2 = np.fft.ifftn(offset_image2)
    offset_image2 = np.array(offset_image2.real)
    offset_image2 = offset_image2.astype("uint8")
        
    M = cv2.getRotationMatrix2D(center, minErrAngleRot1, 1)
    rotated1 = cv2.warpAffine(offset_image2, M, (w, h))
        
    offset_image3 = fourier_shift(np.fft.fftn(rotated1), shift2)
    offset_image3 = np.fft.ifftn(offset_image3)
    offset_image3 = np.array(offset_image3.real)
    offset_image3 = offset_image3.astype("uint8")
        
    M = cv2.getRotationMatrix2D(center, 0, minErrScale1)
    scaledImg1 = cv2.warpAffine(offset_image3, M, (w, h))
    
    return scaledImg1


# %%

###this function will crop, align and convert to HOGs a folder of images, based on specific transformation parameters,
###as output by findAlignParams() - they need to be manually specificed. Useful when parameters have been precalculated,
###for example when aligning a set of recordings of the same mouse, acquired on the same day.
def imagesToHogsCellCropAlign(imgFolderAndFileType, pixelsPerCell, cropCoords, shift1, minErrAngleRot1, shift2, minErrScale1):
    ###this function will crop, align and convert to HOGs a folder of images, based on specific transformation parameters,
    ###as output by findAlignParams() - they need to be manually specificed. Useful when parameters have been precalculated,
    ###for example when aligning a set of recordings of the same mouse, acquired on the same day.
    from skimage import io
    from skimage.feature import hog
    from joblib import Parallel, delayed
    
    coll = io.ImageCollection(imgFolderAndFileType)
    
    if cropCoords == []:
        cropCoords = findCropCoords(imgFolderAndFileType)
    
    r = cropCoords
    
    kwargs = dict(orientations=8, pixels_per_cell=(pixelsPerCell, pixelsPerCell), cells_per_block=(1, 1), transform_sqrt=True)
    return Parallel(n_jobs=30)(delayed(hog)(alignFunc1(image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])], shift1, minErrAngleRot1, shift2, minErrScale1), **kwargs) for image in coll)

# %%

def imagesToHogsCellCropAlignFolder(imageFolder, offset_imageFolder, pixelsPerCell, image_cropCoords, offset_image_cropCoords):
    from skimage import io
    from skimage.feature import hog
    from joblib import Parallel, delayed
    #import cv2
    #import glob
    
    shift1, minErrAngleRot1, shift2, minErrScale1 = findAlignParamsFolder(imageFolder, offset_imageFolder, image_cropCoords, offset_image_cropCoords)
    
    #image = cv2.imread(glob.glob(imageFolder)[1],0)
    #offset_image = cv2.imread(glob.glob(offset_imageFolder)[1],0)
    
    coll = io.ImageCollection(offset_imageFolder)
    
    if image_cropCoords == []:
        image_cropCoords = findCropCoords(offset_imageFolder)
    
    r = image_cropCoords

    tempAlignedIms = Parallel(n_jobs=60)(delayed(alignFunc1)(image, shift1, minErrAngleRot1, shift2, minErrScale1) for image in coll)
    
    kwargs = dict(orientations=8, pixels_per_cell=(pixelsPerCell, pixelsPerCell), cells_per_block=(1, 1), transform_sqrt=True)
    return Parallel(n_jobs=60)(delayed(hog)(image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])], **kwargs) for image in tempAlignedIms)

# %%

def imagesToHogsCellCropAlignFolder2(images, pixelsPerCell, image_cropCoords, shift1, minErrAngleRot1, shift2, minErrScale1):
    from skimage import io
    from skimage.feature import hog
    from joblib import Parallel, delayed
    #import cv2
    #import glob
    
    # coll = io.ImageCollection(offset_imageFolder)
    
    # if image_cropCoords == []:
    #     image_cropCoords = findCropCoords(offset_imageFolder)
    
    r = image_cropCoords

    tempAlignedIms = Parallel(n_jobs=60)(delayed(alignFunc1)(image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])], shift1, minErrAngleRot1, shift2, minErrScale1) for image in images)
    
    kwargs = dict(orientations=8, pixels_per_cell=(pixelsPerCell, pixelsPerCell), cells_per_block=(1, 1), transform_sqrt=True)
    return Parallel(n_jobs=60)(delayed(hog)(image, **kwargs) for image in tempAlignedIms)


# %%

### this function will compare how aligned 2 datasets were before and after alignment and takes images as input
### if alignment has resulted in an improvement, imagesToHogsCellCropAlignFolder() is the recommended follow-up
### if it has not or the difference is small, imagesToHogsCellCrop() is the recommended follow-up since 
### imagesToHogsCellCropAlignFolder() results in a heavy perforance penalty for descriptor creation
def visAlign(image, offset_image, image_cropCoords, offset_image_cropCoords):
    import numpy as np
    import matplotlib.pyplot as plt 
    
    shift1, minErrAngleRot1, shift2, minErrScale1 = findAlignParams(image, offset_image, image_cropCoords, offset_image_cropCoords)
    
    c1 = image_cropCoords
    if c1 != []:
         image = image[c1[1] : c1[1] + c1[3], c1[0] : c1[0] + c1[2]]
    
    c2 = offset_image_cropCoords
    if c2 != []:
         offset_image = offset_image[c2[1] : c2[1] + c2[3], c2[0] : c2[0] + c2[2]]
    
    a2_1 = np.dstack((image, offset_image, np.zeros(image.shape).astype("uint8")))
    a2_2 = np.dstack((image, alignFunc1(offset_image, shift1, minErrAngleRot1, shift2, minErrScale1), np.zeros(image.shape).astype("uint8")))    
    
    f, (ax1, ax2) = plt.subplots(ncols = 2, nrows = 1)
    ax1.imshow(a2_1)
    ax2.imshow(a2_2)
    ax1.axis('off')
    ax1.set_title("original")
    ax2.axis('off')
    ax2.set_title("aligned")
    
    return a2_1, a2_2

# %% 
  
def visAlign_orig(image, offset_image):
    ### this function will compare how aligned 2 datasets were before and after alignment and takes images as input
    ### if alignment has resulted in an improvement, imagesToHogsCellCropAlignFolder() is the recommended follow-up
    ### if it has not or the difference is small, imagesToHogsCellCrop() is the recommended follow-up since 
    ### imagesToHogsCellCropAlignFolder() results in a heavy perforance penalty for descriptor creation
    import numpy as np
    import matplotlib.pyplot as plt 
    
    im0 = image
    im1 = offset_image
    a2_1 = np.dstack((im0, im1, np.zeros(im0.shape).astype("uint8")))
    a2_2 = np.dstack((im0, alignFunc1(im1), np.zeros(im0.shape).astype("uint8")))    
    
    f, (ax1, ax2) = plt.subplots(ncols = 2, nrows = 1)
    ax1.imshow(a2_1)
    ax2.imshow(a2_2)
    ax1.axis('off')
    ax1.set_title("original")
    ax2.axis('off')
    ax2.set_title("aligned")  
# %%

### this function will compare how aligned 2 datasets were before and after alignment and takes folders as input
### if alignment has resulted in an improvement, imagesToHogsCellCropAlignFolder() is the recommended follow-up
### if it has not or the difference is small, imagesToHogsCellCrop() is the recommended follow-up since 
### imagesToHogsCellCropAlignFolder() results in a heavy perforance penalty for descriptor creation
def visAlignFolder(imageFolder, offset_imageFolder, image_cropCoords, offset_image_cropCoords):
    import numpy as np
    import matplotlib.pyplot as plt 
    import cv2
    import glob
    
    image = cv2.imread(glob.glob(imageFolder)[1],0)      
    offset_image = cv2.imread(glob.glob(offset_imageFolder)[1],0)
    
    shift1, minErrAngleRot1, shift2, minErrScale1 = findAlignParams(image, offset_image, image_cropCoords, offset_image_cropCoords)
    
    c1 = image_cropCoords
    if c1 != []:
         image = image[c1[1] : c1[1] + c1[3], c1[0] : c1[0] + c1[2]]
    
    c2 = offset_image_cropCoords
    if c2 != []:
         offset_image = offset_image[c2[1] : c2[1] + c2[3], c2[0] : c2[0] + c2[2]]
    
    a2_1 = np.dstack((image, offset_image, np.zeros(image.shape).astype("uint8")))
    a2_2 = np.dstack((image, alignFunc1(offset_image, shift1, minErrAngleRot1, shift2, minErrScale1), np.zeros(image.shape).astype("uint8")))    
    
    f, (ax1, ax2) = plt.subplots(ncols = 2, nrows = 1)
    ax1.imshow(a2_1)
    ax2.imshow(a2_2)
    ax1.axis('off')
    ax1.set_title("original")
    ax2.axis('off')
    ax2.set_title("aligned")
    
    return a2_1, a2_2

# %%

def visAlignFolder_orig(imageFolder, offset_imageFolder):
    ### this function will compare how aligned 2 datasets were before and after alignment and takes folders as input
    ### if alignment has resulted in an improvement, imagesToHogsCellCropAlignFolder() is the recommended follow-up
    ### if it has not or the difference is small, imagesToHogsCellCrop() is the recommended follow-up since 
    ### imagesToHogsCellCropAlignFolder() results in a heavy perforance penalty for descriptor creation
    import numpy as np
    import matplotlib.pyplot as plt 
    import cv2
    import glob
    
    image = cv2.imread(glob.glob(imageFolder)[1],0)
    offset_image = cv2.imread(glob.glob(offset_imageFolder)[1],0)
    
    shift1, minErrAngleRot1, shift2, minErrScale1 = findAlignParams_orig(image, offset_image)
    
    a2_1 = np.dstack((image, offset_image, np.zeros(image.shape).astype("uint8")))
    a2_2 = np.dstack((image, alignFunc1(offset_image, shift1, minErrAngleRot1, shift2, minErrScale1), np.zeros(image.shape).astype("uint8")))    
    
    f, (ax1, ax2) = plt.subplots(ncols = 2, nrows = 1)
    ax1.imshow(a2_1)
    ax2.imshow(a2_2)
    ax1.axis('off')
    ax1.set_title("original")
    ax2.axis('off')
    ax2.set_title("aligned")

# %%

def findAlignParamsFolder_orig(imageFolder, offset_imageFolder):
    ### this function will attempt to find optimal alignment parameters for a pair of images by employing 
    ### cross-correlation measurement in the fourier space
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import cv2
    import glob

    from skimage.feature import register_translation
    from scipy.ndimage import fourier_shift


    image = cv2.imread(glob.glob(imageFolder)[1],0)
    offset_image = cv2.imread(glob.glob(offset_imageFolder)[1],0)

    # first rought XY alignment
    shift1, error1, diffphase1 = register_translation(image, offset_image)
    
    # apply correction
    offset_image2 = fourier_shift(np.fft.fftn(offset_image), shift1)
    offset_image2 = np.fft.ifftn(offset_image2)
    offset_image2 = np.array(offset_image2.real)
    offset_image2 = offset_image2.astype("uint8")
    
    # get image height, width
    (h, w) = image.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    scale = 1.0
    
    # find rotational alignment
    rotAngle1 = []
    rotError1 = []
    for i in range(-45,45):
        M = cv2.getRotationMatrix2D(center, i, 1)
        rotated_i = cv2.warpAffine(offset_image2, M, (w, h))
        
        shiftRot1, errorRot1, diffphaseRot1 = register_translation(image, rotated_i)
        
        rotAngle1.append(i)
        rotError1.append(errorRot1)
    
    # detect angle with the lowest error
    rot1_df = [rotAngle1, rotError1]
    rot1_df = pd.DataFrame(rot1_df)
    rot1_df = rot1_df.T
    minErrRowRot1 = rot1_df[1].idxmin()
    minErrAngleRot1 = rot1_df[0][minErrRowRot1]
    
    # apply rotational correction
    M = cv2.getRotationMatrix2D(center, minErrAngleRot1, 1)
    rotated1 = cv2.warpAffine(offset_image2, M, (w, h))
    
    
    # check for XY alignment again
    shift2, error2, diffphase2 = register_translation(image, rotated1)
    
    # apply XY correction again
    offset_image3 = fourier_shift(np.fft.fftn(rotated1), shift2)
    offset_image3 = np.fft.ifftn(offset_image3)
    offset_image3 = np.array(offset_image3.real)
    offset_image3 = offset_image3.astype("uint8")
    
    
    
    # check for scale alignment
    scale1 = []
    scaleError1 = []
    for i in np.arange(0.0, 2.0, 0.01):
        M = cv2.getRotationMatrix2D(center, 0, i)
        rescaled_i = cv2.warpAffine(offset_image3, M, (w, h))
        
        shiftScale1, errorScale1, diffphaseScale1 = register_translation(image, rescaled_i)
        
        scale1.append(i)
        scaleError1.append(errorScale1)
        
    scale1_df = [scale1, scaleError1]
    scale1_df = pd.DataFrame(scale1_df)
    scale1_df = scale1_df.T
    minErrRowScale1 = scale1_df[1].idxmin()
    minErrScale1 = scale1_df[0][minErrRowScale1]
    
    M = cv2.getRotationMatrix2D(center, 0, minErrScale1)
    scaledImg1 = cv2.warpAffine(offset_image3, M, (w, h))    
    
    
    return shift1, minErrAngleRot1, shift2, minErrScale1

# %%


### computes variance explained by a PCA model on sepcified data
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


