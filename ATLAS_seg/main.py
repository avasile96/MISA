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
import itk
from itkwidgets import compare, checkerboard, view

im_f_path = 'C:\\Users\\usuari\\Javascript_API\\MISA\\ATLAS_seg\\testing-set\\testing-images\\1003.nii.gz' # TO_DO
im_m_path = 'C:\\Users\\usuari\\Javascript_API\\MISA\\ATLAS_seg\\testing-set\\testing-images\\1004.nii.gz' # TO_DO

# Reading images
im_f = itk.imread(im_f_path, itk.F)
im_m = itk.imread(im_m_path, itk.F)

# Initialization
parameter_object = itk.ParameterObject.New()
default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('bspline')
parameter_object.AddParameterMap(default_rigid_parameter_map)

# Call registration function
result_image, result_transform_parameters = itk.elastix_registration_method(
    im_f, im_m,
    parameter_object=parameter_object,
    log_to_console=False)


# Load Elastix Image Filter Object
elastix_object = itk.ElastixRegistrationMethod.New(im_f, im_m)
# elastix_object.SetFixedImage(fixed_image)
# elastix_object.SetMovingImage(moving_image)
elastix_object.SetParameterObject(parameter_object)

# Set additional options
elastix_object.SetLogToConsole(False)

# Update filter object (required)
elastix_object.UpdateLargestPossibleRegion()

# Results of Registration
result_image = elastix_object.GetOutput()
result_transform_parameters = elastix_object.GetTransformParameterObject()

# Comparing the Results
# a = checkerboard(im_f, result_image)
# b = compare(im_f, result_image, link_cmap=True)
dif_og = np.subtract(im_f, im_m)
print(dif_og)
dif_reg = np.subtract(im_f, result_image)
print(sum(sum(dif_reg)))
             
# Save image with itk
itk.imwrite(result_image,'1004_reg.nii.gz')


