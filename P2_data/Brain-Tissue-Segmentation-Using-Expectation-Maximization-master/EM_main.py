# -*- coding: utf-8 -*-
"""
Created on Thu Oct  21 06:45:33 2021

@authors: Manuel Ojeda & Alexandru Vasile

This implementation only works for 3 clusters (due to time constrictions)
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import time
import operator
from math import sqrt, pi
from numpy.linalg import inv, det, norm
from functools import partial
from scipy.spatial.distance import dice

# Showing the 2d slices
def show_slice(img, slice_nr):
    plt.figure()
    plt.imshow(img[:,:,slice_nr].T, cmap='gray')


def GaussMixModel(x, mean, cov):
    """
    In:
        x (np array): nxd dimentional, n= number of samples; d= dimention
        mean (np array): d-dimentional, mean value.
        cov (np array): dxd dimentional covariance matrix.
    Out:
        gaus_mix_model (np array): Gaussian mixture for every point in feature space.
    """
    gaus_mix_model = np.exp(-0.5*(x - mean) @ inv(cov) @ np.transpose(x - mean)) / (2 * pi * sqrt(det(cov)))
    
    return gaus_mix_model



def DICE(Seg_img, GT_img,state):
    """   
    In:
        Seg_img (np array): Segmented Image.
        GT_img (np array): Ground Truth Image.
        State: "nifti" if the images are nifti file
               "arr"   if the images are an ndarray
    out:
        Dice Similarity Coefficient: dice_CSF, dice_GM, dice_WM."""
        
    import numpy as np
    if (state=="nifti"):
       segmented_data = Seg_img.get_data().copy()
       groundtruth_data = GT_img.get_data().copy()
    elif (state=="arr"):
       segmented_data = Seg_img.copy()
       groundtruth_data = GT_img.copy()
    
    #Calculte DICE
    def dice_coefficient(SI,GT):
        #   2 * TP / (FN + (2 * TP) + FP)
        intersection = np.logical_and(SI, GT)
        return 2. * intersection.sum() / (SI.sum() + GT.sum())
    
    #Dice  for CSF
    Seg_CSF = (segmented_data == 1) * 1
    GT_CSF = (groundtruth_data == 1) * 1
    dice_CSF = dice_coefficient(Seg_CSF, GT_CSF)
    #Dice  for GM
    Seg_GM = (segmented_data == 2) * 1
    GT_GM = (groundtruth_data == 2) * 1
    dice_GM = dice_coefficient(Seg_GM, GT_GM)
    #Dice  for WM
    Seg_WM = (segmented_data == 3) * 1
    GT_WM = (groundtruth_data == 3) * 1
    dice_WM = dice_coefficient(Seg_WM, GT_WM)
    
    return dice_CSF, dice_GM, dice_WM

def Dice_and_Visualization_of_one_slice(Seg_img, GT_img,state, slice_nr):
    """      
     In:
        Seg_img (np array): Segmented Image.
        GT_img (np array): Ground Truth Image.
        State: "nifti" if the images are nifti file
               "arr"   if the images are an ndarray
    out:
        Dice Similarity Coefficient: dice_CSF, dice_GM, dice_WM.
        Ploting image:"""
    
    import numpy as np
    if (state=="nifti"):
       segmented_data = Seg_img.get_data().copy()
       groundtruth_data = GT_img.get_data().copy()
    elif (state=="arr"):
       segmented_data = Seg_img.copy()
       groundtruth_data = GT_img.copy()
    
    #Calculte DICE
    def dice_coefficient(SI,GT):
        #   2 * TP / (FN + (2 * TP) + FP)
        intersection = np.logical_and(SI, GT)
        return 2. * intersection.sum() / (SI.sum() + GT.sum())
    
    #Dice  for CSF
    Seg_CSF = (segmented_data == 1) * 1
    GT_CSF = (groundtruth_data == 1) * 1
    dice_CSF = dice_coefficient(Seg_CSF, GT_CSF)
    #Dice  for GM
    Seg_GM = (segmented_data == 2) * 1
    GT_GM = (groundtruth_data == 2) * 1
    dice_GM = dice_coefficient(Seg_GM, GT_GM)
    #Dice  for WM
    Seg_WM = (segmented_data == 3) * 1
    GT_WM = (groundtruth_data == 3) * 1
    dice_WM = dice_coefficient(Seg_WM, GT_WM)
    
    print("CSF DICE = {}".format(dice_CSF), "GM DICE = {}".format(dice_GM), "WM DICE = {}".format(dice_WM))
    
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,figsize=(12,8))
    
    ax1.set_title("WM Segmentation of slice no. {}".format(slice_nr))
    img1 = ax1.imshow(Seg_WM[:,:,slice_nr].T, cmap = "gray")
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    
    ax2.set_title("CSF Segmentation of slice no. {}".format(slice_nr))
    img2 = ax2.imshow(Seg_CSF[:,:,slice_nr].T, cmap = "gray")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    ax3.set_title("GM Segmentation of sliceno. {}".format(slice_nr))
    img3 = ax3.imshow(Seg_GM[:,:,slice_nr].T, cmap = "gray")
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)
    
    ax4.set_title("WM Ground Truth of slice no. {}".format(slice_nr))
    img4 = ax4.imshow(GT_WM[:,:,slice_nr].T, cmap = "gray")
    ax4.axes.get_xaxis().set_visible(False)
    ax4.axes.get_yaxis().set_visible(False)

    ax5.set_title("CSF Segmentation of slice no. {}".format(slice_nr))
    img5 = ax5.imshow(GT_CSF[:,:,slice_nr].T, cmap = "gray")
    ax5.axes.get_xaxis().set_visible(False)
    ax5.axes.get_yaxis().set_visible(False)

    ax6.set_title("GM Ground Truth of slice no. {}".format(slice_nr))
    img6 = ax6.imshow(GT_GM[:,:,slice_nr].T, cmap = "gray")
    ax6.axes.get_xaxis().set_visible(False)
    ax6.axes.get_yaxis().set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
####################### MAIN #############################
slice_nr = 20
############## Loading data ###################
brain_data_path ="./P2_data/2" # indicating data location

# Load T1_image
T1_data = os.path.join(brain_data_path, 'T1.nii')
T1_data = nib.load(T1_data)
T1_img=T1_data.get_fdata()

# Load T2_Flair_image
T2_data = os.path.join(brain_data_path, 'T2_FLAIR.nii')
T2_data = nib.load(T2_data)
T2_img = T2_data.get_fdata()

# Load Label Image
labeled_data = os.path.join(brain_data_path, 'LabelsForTesting.nii')
labeled_data = nib.load(labeled_data)
labeled_img = labeled_data.get_fdata()

# Ploting images  
show_slice(labeled_img, 24)
show_slice(T1_img, 24)
show_slice(T2_img, 24)


### Selecting the Region of Interest (ROI) ###
"""
    Multyplying the labeled image with the working images of the project
    to only work on the required region
 """

label_copy = labeled_img.copy()
label_copy[label_copy>0] = 1 # safety measure
# multiplying with working volume
T1_masked = np.multiply(T1_img, label_copy)
T2_masked = np.multiply(T2_img, label_copy)
# saving the masked ROI images
T1_ROI_data = nib.Nifti1Image(T1_masked, T1_data.affine, T1_data.header)
T1_ROI = T1_ROI_data.get_fdata()

T2_ROI_data = nib.Nifti1Image(T2_masked, T2_data.affine, T2_data.header)
T2_ROI = T2_ROI_data.get_fdata()
# imshowing the slices
show_slice(label_copy, slice_nr)
show_slice(labeled_img, slice_nr)
show_slice(T1_ROI, slice_nr)
show_slice(T2_ROI, slice_nr)

# Vectorizing the images
T1_flat = T1_ROI.copy().flatten()
T2_flat = T2_ROI.copy().flatten()

T1T2_linear_stack = np.vstack((T1_flat, T2_flat)).T
# Getting the nonzero elements of the working object
"""
enumerate:It allows us to loop over something and have an automatic counter
"""
T1T2_stack_nnz_x_index = [i for i, x in enumerate(T1T2_linear_stack) if x.any()] # indices of nnz elements
T1T2_stack_nnz = T1T2_linear_stack[T1T2_stack_nnz_x_index] # selecting only nonzero elements

#################### Initialization ######################

######### Kmeans Initialization ###############
"""
In: n_clusters = Number of cluster
       K-means++: Centroid initialization for faster convergence
       random_state: For reproducibility
                     42 in this case because it is the answer to everything!
Out:
      Kmeans_predict = cluster segmentation labels
      Centroid = centroids determined by the clustering
""" 
kmeans=KMeans(n_clusters=3,  init='k-means++',random_state=42,).fit(T1T2_stack_nnz)
Kmeans_pred=kmeans.predict(T1T2_stack_nnz)
centroids = kmeans.cluster_centers_

"""
    Making the Kmeans output the same vlaues for the outputted labels.
    Without this, it will give seemengly random numbers every time.
"""

# finding the minimum and maximum values of labels
y_centroids = centroids[:,0]
# min and max label indices and their values
min_index, min_value = min(enumerate(y_centroids), key=operator.itemgetter(1))
max_index, max_value = max(enumerate(y_centroids), key=operator.itemgetter(1))

# creating an empty variable for the new clustering arrangement

Kmeans_pred_new=np.zeros(len(Kmeans_pred))
centroid_new=np.zeros(centroids.shape)

# centrioid arrangement
centroid_new[0]=centroids[min_index]
centroid_new[2]=centroids[max_index]
# rearranging the centrioid
if (min_index+max_index==1):
   centroid_new[1]=centroids[2]
elif (min_index+max_index==2):
   centroid_new[1]=centroids[1]
elif (min_index+max_index==3):
   centroid_new[1]=centroids[0]

# new labels
for i in range(0,len(Kmeans_pred)):
    if (Kmeans_pred[i]==min_index):
        Kmeans_pred_new[i]=0
    elif(Kmeans_pred[i]==max_index):
        Kmeans_pred_new[i]=2
    else:
        Kmeans_pred_new[i]=1

Kmeans_pred_new += 1 # just to start with 1 instead of 0

# getting the label region images  

"""
   argwhere(Kmeans_pred_new == 1)[:,0]: Go through the predictions of the 
   kmeans and get corresponding indices
"""    
CSF_stack = T1T2_stack_nnz[np.argwhere(Kmeans_pred_new == 1)[:,0],:]
GM_stack = T1T2_stack_nnz[np.argwhere(Kmeans_pred_new== 2)[:,0],:]
WM_stack = T1T2_stack_nnz[np.argwhere(Kmeans_pred_new== 3)[:,0],:]

"""
    np.mean(X, axis = 0): Compute the mean along colums. means: along features, mean of feature.
""" 
# computing means and coveriences of each region
mean_CSF = np.mean(CSF_stack, axis = 0)
cov_CSF = np.cov(CSF_stack, rowvar = False)
mean_GM = np.mean(GM_stack, axis = 0)
cov_GM = np.cov(GM_stack, rowvar = False)
mean_WM = np.mean(WM_stack , axis = 0)
cov_WM = np.cov(WM_stack , rowvar = False)

# Prior_Probabibilities
pp_CSF = CSF_stack.shape[0] / T1T2_stack_nnz.shape[0]
pp_GM = GM_stack.shape[0] / T1T2_stack_nnz.shape[0]
pp_WM = WM_stack.shape[0] / T1T2_stack_nnz.shape[0]

##Ploting the cluster distributin    
plt.figure()
plt.scatter(T1T2_stack_nnz[:, 0], T1T2_stack_nnz[:, 1], c=Kmeans_pred_new, s=25)
plt.scatter(centroid_new[:, 0], centroid_new[:, 1], marker='x', s=200, linewidths=3, color='w', zorder=10)
plt.show()


######################## EM algorithm ############################

MAX_STEPS = 3
min_err = 1e-3 # 0.001
n_steps = 0
label_distribution = np.array((pp_CSF, pp_GM , pp_WM))

fig=plt.figure()

while True:
    
### EXPECTATION STEP ###
   CSF_gmm= np.apply_along_axis(partial(GaussMixModel, mean=mean_CSF, cov=cov_CSF), 1, T1T2_stack_nnz)
   GM_gmm= np.apply_along_axis(partial(GaussMixModel, mean=mean_GM, cov=cov_GM), 1, T1T2_stack_nnz)
   WM_gmm= np.apply_along_axis(partial(GaussMixModel, mean=mean_WM, cov=cov_WM), 1, T1T2_stack_nnz)
   
   ## constructing the weights (formula from slides)
   pp_x_gmm= (pp_CSF*CSF_gmm)+(pp_GM*GM_gmm)+(pp_WM*WM_gmm) # Denominator
   
   # numerators: 1 for each label
   weights_CSF=(pp_CSF*CSF_gmm)/pp_x_gmm
   weights_GM=(pp_GM*GM_gmm)/pp_x_gmm
   weights_WM=(pp_WM*WM_gmm)/pp_x_gmm

   weights=np.vstack((weights_CSF,weights_GM,weights_WM))
   weights=np.transpose(weights)
   
   # Metric (old)
   log_o=sum((np.log(sum(weights))))
   
### MAXIMIZATION STEP ###
   Prediction=np.argmax(weights,axis=1)
   Prediction=Prediction+1

   _,counts = np.unique(Prediction, return_counts=True)
   pp_CSF = counts[0] / T1T2_stack_nnz.shape[0]
   pp_GM = counts[1] / T1T2_stack_nnz.shape[0]
   pp_WM = counts[2] / T1T2_stack_nnz.shape[0]

   label_distribution_new = np.array((pp_CSF, pp_GM , pp_WM))

### calculating new means and covariances ###
   mean_CSF= (1/counts[0]) * (weights[:, 0] @ T1T2_stack_nnz)
   mean_GM= (1/counts[1]) * (weights[:, 1] @ T1T2_stack_nnz)
   mean_WM= (1/counts[2]) * (weights[:, 2] @ T1T2_stack_nnz)
   cov_CSF = (1/counts[0]) * (weights[:, 0] * np.transpose(T1T2_stack_nnz - mean_CSF)) @ (T1T2_stack_nnz - mean_CSF)
   cov_GM= (1/counts[1]) * (weights[:, 1] * np.transpose(T1T2_stack_nnz - mean_GM)) @ (T1T2_stack_nnz - mean_GM)
   cov_WM= (1/counts[2]) * (weights[:, 2] * np.transpose(T1T2_stack_nnz - mean_WM)) @ (T1T2_stack_nnz - mean_WM)

### Generating new GMMs ##
   CSF_gmm= np.apply_along_axis(partial(GaussMixModel, mean=mean_CSF, cov=cov_CSF), 1, T1T2_stack_nnz)
   GM_gmm= np.apply_along_axis(partial(GaussMixModel, mean=mean_GM, cov=cov_GM), 1, T1T2_stack_nnz)
   WM_gmm= np.apply_along_axis(partial(GaussMixModel, mean=mean_WM, cov=cov_WM), 1, T1T2_stack_nnz)

   pp_x_gmm= (pp_CSF*CSF_gmm)+(pp_GM*GM_gmm)+(pp_WM*WM_gmm)

   weights_CSF=(pp_CSF*CSF_gmm)/pp_x_gmm
   weights_GM=(pp_GM*GM_gmm)/pp_x_gmm
   weights_WM=(pp_WM*WM_gmm)/pp_x_gmm

   weights=np.vstack((weights_CSF,weights_GM,weights_WM))
   weights=np.transpose(weights)
   
   # New Metric (new)
   log_n=sum((np.log(sum(weights))))
   
   
   ditribution_difference = norm(label_distribution_new - label_distribution)
   log_err=  norm(log_n-log_o)
   print("-------------------------------------")
   print("Step %d" % n_steps)
   print("Distribution change %f" % ditribution_difference)
   print("log change %f" %  log_err)
   n_steps += 1

    # Stopping the WHILE --> stopping criteria:
    # - max number of iterations has been reached
    # or
    # - the desired metric error has been reached

   if (n_steps >= MAX_STEPS) or (log_err <= min_err):
       print("Maximization Completed")
       break
   else:
        label_distribution = label_distribution_new


################# Recontructing the image ############################

seg_vector = np.zeros_like(T2_flat)
seg_vector[T1T2_stack_nnz_x_index] = Prediction
seg_img = np.reshape(seg_vector,T1_img.shape)

show_slice(label_copy, slice_nr)
show_slice(labeled_img, slice_nr)
show_slice(T1_ROI, slice_nr)
show_slice(seg_img, slice_nr)

show_slice(label_copy, slice_nr)
show_slice(labeled_img, slice_nr)
show_slice(T1_ROI, slice_nr)
show_slice(seg_img, slice_nr)

################## DICE Coefficient ##############################
dice_CSF, dice_GM, dice_WM = DICE(seg_img,labeled_img,"arr")
print("CSF DICE = {}".format(dice_CSF), "GM DICE = {}".format(dice_GM), "WM DICE = {}".format(dice_WM))

# Plotting all labels together in one slice
plt.figure()
plt.imshow(seg_img[:,:,slice_nr].T, cmap='plasma')

# Plotting label segmentation along with Ground Truth  
Dice_and_Visualization_of_one_slice(seg_img,labeled_img,"arr",slice_nr)