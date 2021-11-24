# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 18:06:35 2021

@author: vasil

DICE Calculation
"""

# Libraries
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from prime_aux import computeAtlasProb, computeTissueModels, labelPropg, all_labelPropg, computeMNIAtlasProb
from em_aux import dice_metric
from EM_main_final import segmentEM, DICE


indexes = ['003', '004', '005', '018', '019', '023', '024', '025', 
                 '038', '039', '101', '104', '104', '107', '110', '113', 
                 '116', '119', '122', '125', '128']

# Label Propagation Segmentation
for index in indexes:
    label_CSF_dir = "../results/testing_results/transformed_labels/CSF/"+index+"/result.mhd"
    label_GM_dir = "../results/testing_results/transformed_labels/WM/"+index+"/result.mhd"
    label_WM_dir = "../results/testing_results/transformed_labels/WM/"+index+"/result.mhd"
    
    lab_img_csf = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label_CSF_dir)))
    lab_img_gm = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label_GM_dir)))
    lab_img_wm = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label_WM_dir)))
    
    lab_img_gm = lab_img_gm * 2
    lab_img_wm = lab_img_wm * 3
    
    seg_img = lab_img_csf + lab_img_gm + lab_img_wm
    
    mask_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz"
    ground_truth = np.array(sitk.GetArrayFromImage(sitk.ReadImage(mask_dir)))
    
    dice_csf, _, _ = DICE(lab_img_csf, ground_truth, imtype="arr")
    _, dice_gm, _ = DICE(lab_img_gm, ground_truth, imtype="arr")
    _, _, dice_wm = DICE(lab_img_wm, ground_truth, imtype="arr")
    print(dice_csf, dice_gm, dice_wm)
    
# Label Propagation average atlas
for index in indexes:
    label_CSF_dir = "../results/testing_results/transformed_labels_avg/CSF/"+index+"/result.mhd"
    label_GM_dir = "../results/testing_results/transformed_labels_avg/WM/"+index+"/result.mhd"
    label_WM_dir = "../results/testing_results/transformed_labels_avg/WM/"+index+"/result.mhd"
    
    lab_img_csf = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label_CSF_dir)))
    lab_img_gm = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label_GM_dir)))
    lab_img_wm = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label_WM_dir)))
    
    lab_img_gm = lab_img_gm * 2
    lab_img_wm = lab_img_wm * 3
    
    seg_img = lab_img_csf + lab_img_gm + lab_img_wm
    
    mask_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz"
    ground_truth = np.array(sitk.GetArrayFromImage(sitk.ReadImage(mask_dir)))
    
    dice_csf, _, _ = DICE(lab_img_csf, ground_truth, imtype="arr")
    _, dice_gm, _ = DICE(lab_img_gm, ground_truth, imtype="arr")
    _, _, dice_wm = DICE(lab_img_wm, ground_truth, imtype="arr")
    print(dice_csf, dice_gm, dice_wm)
    
# Label Propagation MNI
for index in indexes:
    label_CSF_dir = "../results/testing_results/transformed_labels_MNI/CSF/"+index+"/result.mhd"
    label_GM_dir = "../results/testing_results/transformed_labels_MNI/WM/"+index+"/result.mhd"
    label_WM_dir = "../results/testing_results/transformed_labels_MNI/WM/"+index+"/result.mhd"
    
    lab_img_csf = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label_CSF_dir)))
    lab_img_gm = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label_GM_dir)))
    lab_img_wm = np.array(sitk.GetArrayFromImage(sitk.ReadImage(label_WM_dir)))
    
    lab_img_gm = lab_img_gm * 2
    lab_img_wm = lab_img_wm * 3
    
    seg_img = lab_img_csf + lab_img_gm + lab_img_wm
    
    mask_dir="../data/testing-set/testing-labels/1"+index+"_3C.nii.gz"
    ground_truth = np.array(sitk.GetArrayFromImage(sitk.ReadImage(mask_dir)))
    
    dice_csf, _, _ = DICE(lab_img_csf, ground_truth, imtype="arr")
    _, dice_gm, _ = DICE(lab_img_gm, ground_truth, imtype="arr")
    _, _, dice_wm = DICE(lab_img_wm, ground_truth, imtype="arr")
    print(dice_csf, dice_gm, dice_wm)