import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import scipy.stats
import math
import cv2
from scipy import ndimage

from sklearn import cluster
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
from skimage import io
from EM import read_img, flatten_img, EM_cluster, recover_img, ex



FILENAME1 = './img/john-lawrence-sullivan-aw18.jpg'
FILENAME2 = './img/Co6q6Fry23KAGdrRAAJBxxyYEHQ922.jpg'

orig_img = read_img(FILENAME1)
x, y, z = orig_img.shape
plt.figure()
plt.axis("off")
plt.imshow(orig_img)
plt.title('Original Image');
plt.show()

img = flatten_img(orig_img)
labels, means, cov, pis, likelihood_arr = EM_cluster(img, 5)
em_img = recover_img(means[labels], X=x, Y=y, Z=z)
plt.figure()
plt.axis("off")
plt.imshow(em_img)
plt.title('Image Segmented by EM Algorithm');
plt.show()

print(ex)