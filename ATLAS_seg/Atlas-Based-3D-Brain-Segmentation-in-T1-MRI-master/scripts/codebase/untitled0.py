# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 23:17:26 2021

@author: vasil
"""

import numpy as np
import SimpleITK as sitk
import warnings
from skimage import io
warnings.filterwarnings("ignore")

CSF = 'D:\\Uni\\Spain\\MIRA\\dataset\\results\\result.mha'

a = np.array(sitk.GetArrayFromImage(sitk.ReadImage(CSF)))
output_prediction        = sitk.GetImageFromArray(a)
writer           = sitk.ImageFileWriter()
writer.SetFileName('D:\\Uni\\Spain\\MIRA\\dataset\\results\\a.tif')
writer.Execute(output_prediction)
