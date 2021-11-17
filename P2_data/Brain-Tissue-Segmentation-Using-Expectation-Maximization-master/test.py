"""
Created on Thu Oct  21 06:45:33 2021

@authors: Manuel Ojeda & Alexandru Vasile

This implementation only works for 3 clusters
but can be modified for more if need be.
"""

import numpy as np
import nibabel as nib
import operator
from math import sqrt, pi
from numpy.linalg import inv, det, norm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
from functools import partial
import time


# Showing the 2d slices
def show_slice(img, slice_nr):
    plt.figure()
    plt.imshow(img[:,:,slice_nr].T, cmap='gray')


def GaussMixModel(x, mean, cov):
    """
    In:
        x (np array): nxd dimentional, n= number of samples; d = dimention
        mean (np array): d-dimentional, mean value.
        cov (np array): dxd dimentional covariance matrix.
    Out:
        gaus_mix_model (np array): Gaussian mixture for every point in feature space.
    """
    gaus_mix_model = np.exp(-0.5*(x - mean) @ cov**-1 @ np.transpose(x - mean)) / (2 * pi * sqrt(cov[0,0]))
    
    return gaus_mix_model


def dice_itself(segmentation, ground_truth):
        and_gate = np.logical_and(segmentation, ground_truth)
        return 2. * and_gate.sum() / (segmentation.sum() + ground_truth.sum())
    
def DICE(seg_im, ground_truth, imtype):
    """   
    In:
        seg_im (np array): Segmented Image.
        ground_truth (np array): Ground Truth Image.
        State: "nifti" if the images are nifti file
               "arr"   if the images are an ndarray
    out:
        Dice Similarity Coeff. for each tissue: dice_csf, dice_gm, dice_wm."""

    if (imtype=="nifti"):
       seg_data = seg_im.get_data().copy()
       groundtruth_data = ground_truth.get_data().copy()
    elif (imtype=="arr"):
       seg_data = seg_im.copy()
       groundtruth_data = ground_truth.copy()
    
    seg_csf = (seg_data == 1) * 1
    gt_csf = (groundtruth_data == 1) * 1
    dice_csf = dice_itself(seg_csf, gt_csf)

    seg_gm = (seg_data == 2) * 1
    gt_gm = (groundtruth_data == 2) * 1
    dice_gm = dice_itself(seg_gm, gt_gm)

    seg_wm = (seg_data == 3) * 1
    gt_wm = (groundtruth_data == 3) * 1
    dice_wm = dice_itself(seg_wm, gt_wm)
    
    return dice_csf, dice_gm, dice_wm

def Slice_and_Dice(seg_im, ground_truth, imtype, slice_nr):
    """
    Computes the DICE coefficients for the classes of just one slice.
    """

    if (imtype=="nifti"):
       seg_data = seg_im.get_data().copy()
       groundtruth_data = ground_truth.get_data().copy()
    elif (imtype=="arr"):
       seg_data = seg_im.copy()
       groundtruth_data = ground_truth.copy()
    

    seg_csf = (seg_data == 1) * 1
    gt_csf = (groundtruth_data == 1) * 1
    dice_csf = dice_itself(seg_csf[:,:,slice_nr], gt_csf[:,:,slice_nr])

    seg_gm = (seg_data == 2) * 1
    gt_gm = (groundtruth_data == 2) * 1
    dice_gm = dice_itself(seg_gm[:,:,slice_nr], gt_gm[:,:,slice_nr])

    seg_wm = (seg_data == 3) * 1
    gt_wm = (groundtruth_data == 3) * 1
    dice_wm = dice_itself(seg_wm[:,:,slice_nr], gt_wm[:,:,slice_nr])
    
    print("CSF DICE = {}".format(dice_csf), "GM DICE = {}".format(dice_gm), "WM DICE = {}".format(dice_wm))

    
def init(init_type):
    """
    In: init_type: 'kmeans', 'random'
    Out:
          Initial parameters for the EM algorithm
    """ 
    if (init_type =='kmeans'):
        ### Kmeans Initialization
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
        
        # Prior Probabibilities (How likely are we to encounter that type of tissue)
        pp_CSF = CSF_stack.shape[0] / T1T2_stack_nnz.shape[0]
        pp_GM = GM_stack.shape[0] / T1T2_stack_nnz.shape[0]
        pp_WM = WM_stack.shape[0] / T1T2_stack_nnz.shape[0]
        
    else:
        ### Random Initialization
        rand_init_vect = np.random.randint(1,4,T1T2_stack_nnz.shape[0])
        
        # getting the label region images
        CSF_stack = T1T2_stack_nnz[np.argwhere(rand_init_vect == 1)[:,0],:]
        GM_stack = T1T2_stack_nnz[np.argwhere(rand_init_vect== 2)[:,0],:]
        WM_stack = T1T2_stack_nnz[np.argwhere(rand_init_vect== 3)[:,0],:]
        
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
        
    return pp_CSF, pp_GM, pp_WM, mean_CSF, cov_CSF, mean_GM, cov_GM, mean_WM, cov_WM
    
####################### MAIN #############################
slice_nr = 20
############## Loading data ###################
brain_data_path ="./P2_data/2" # indicating data location

# Load T1 image
T1_data = os.path.join(brain_data_path, 'T1.nii')
T1_data = nib.load(T1_data)
T1_img=T1_data.get_fdata()

# Load T2_Flair image
T2_data = os.path.join(brain_data_path, 'T2_FLAIR.nii')
T2_data = nib.load(T2_data)
T2_img = T2_data.get_fdata()

# Load Label Image
labeled_data = os.path.join(brain_data_path, 'LabelsForTesting.nii')
labeled_data = nib.load(labeled_data)
labeled_img = labeled_data.get_fdata()

# Ploting images  
show_slice(labeled_img, slice_nr)
show_slice(T1_img, slice_nr)
show_slice(T2_img, slice_nr)


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

T2_ROI_data = nib.Nifti1Image(T1_masked, T1_data.affine, T1_data.header)
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

t0 = time.time()

# Initialization (random or kmeans)
init_type = 'random'
pp_CSF, pp_GM, pp_WM, mean_CSF, cov_CSF, mean_GM, cov_GM, mean_WM, cov_WM = init(init_type) #kmeans #random

######################## EM algorithm ############################
MAX_STEPS = 3
min_err = 1e-3 # 0.001
n_steps = 0
label_distribution = np.array((pp_CSF, pp_GM , pp_WM))

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
        
t1 = time.time() # time of convergence
################# Recontructing the image ############################

seg_vector = np.zeros_like(T2_flat)
seg_vector[T1T2_stack_nnz_x_index] = Prediction
seg_img = np.reshape(seg_vector,T1_img.shape)

################ PLOTTING #####################

show_slice(label_copy, slice_nr) # ROI
show_slice(labeled_img, slice_nr) # og labels
show_slice(T1_ROI, slice_nr) # ROI*T1
show_slice(seg_img, slice_nr) # our segmentation

# Plotting label segmentation along with Ground Truth  
Slice_and_Dice(seg_img,labeled_img,"arr",slice_nr)
print("=========================================================")

################## DICE Coefficient ##############################
dice_csf, dice_gm, dice_wm = DICE(seg_img,labeled_img,"arr")
print("CSF DICE (slice no. {}) = {}".format(slice_nr, dice_csf), "GM DICE (slice no. {})= {}".format(slice_nr, dice_gm), "WM DICE = (slice no. {}){}".format(slice_nr, dice_wm))

################## Time Elapsed ##############################
print("Initialization type = {};".format(init_type), "Time elapsed = {}".format(t1-t0))