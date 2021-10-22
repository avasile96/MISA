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


# input fllename >> output 3d array
def read_img(filename):
    img_3d = io.imread(filename)

    small = cv2.resize(img_3d, (0, 0), fx=0.1, fy=0.1)
    blur = cv2.blur(small, (4, 4))
    return blur


# input 3d array >> output 2d array
def flatten_img(img_3d):
    z, x, y = img_3d.shape
    img_2d = img_3d.reshape(x*y, z)
    img_2d = np.array(img_2d, dtype = np.float)
    return img_2d


# input 2d array >> output 3d array
def recover_img(img_2d, vis = False, X=80, Y=80, Z=3):
    #img_2d = cv2.resize(img_2d, (0, 0), fx=10, fy=10)
    img_2d = (img_2d * 255).astype(np.uint8)
    recover_img = img_2d.reshape(X, Y, Z)
    return recover_img


# input 2d array >> output estimated means, stds, pis
def kmeans_init(img, k):
    means, labels = kmeans2(img, k)
    try:
        means = np.array(means) #mean
        cov = np.array([np.cov(img[labels == i].T) for i in range(k)]) #covariance
        ids = set(labels) #labels
        pis = np.array([np.sum([labels == i]) / len(labels) for i in ids]) # general probability of a pixel belonging to a cluster
    except Exception as ex:
        pass
    return means, cov, pis


# E-Step: Update Parameters
# update the conditional pdf - prob that pixel i given class j
def update_responsibility(img, means, cov, pis, k):
    # responsibilities: i th pixels, j th class
    # pis * gaussian.pdf
    responsibilities = np.array([pis[j] * scipy.stats.multivariate_normal.pdf(img, mean=means[j], cov=cov[j]) for j in range(k)]).T
    # normalize for each row
    norm = np.sum(responsibilities, axis = 1)
    # convert to column vector
    norm = np.reshape(norm, (len(norm), 1))
    responsibilities = responsibilities / norm
    return responsibilities


# update pi for each class of Gaussian model
def update_pis(responsibilities):
    pis = np.sum(responsibilities, axis = 0) / responsibilities.shape[0]
    return pis


# update means for each class of Gaussian model
def update_means(img, responsibilities):
    means = []
    class_n = responsibilities.shape[1]
    for j in range(class_n):
        weight = responsibilities[:, j] / np.sum(responsibilities[:, j])
        weight = np.reshape(weight, (1, len(weight)))
        means_j = weight.dot(img)
        means.append(means_j[0])
    means = np.array(means)
    return means


# update covariance matrix for each class of Gaussian model
def update_covariance(img, responsibilities, means):
    cov = []
    class_n = responsibilities.shape[1]
    for j in range(class_n):
        weight = responsibilities[:, j] / np.sum(responsibilities[:, j])
        weight = np.reshape(weight, (1, len(weight)))
        # Each pixels have a covariance matrice
        covs = [np.mat(i - means[j]).T * np.mat(i - means[j]) for i in img]
        # Weighted sum of covariance matrices
        cov_j = sum(weight[0][i] * covs[i] for i in range(len(weight[0])))
        cov.append(cov_j)
    cov = np.array(cov)
    return cov


# M-step: choose a label that maximise the likelihood
def update_labels(responsibilities):
    labels = np.argmax(responsibilities, axis = 1)
    return labels


def update_loglikelihood(img, means, cov, pis, k):
    pdf = np.array([pis[j] * scipy.stats.multivariate_normal.pdf(img, mean=means[j], cov=cov[j]) for j in range(k)])
    log_ll = np.log(np.sum(pdf, axis = 0))
    log_ll_sum = np.sum(log_ll)
    return log_ll_sum


def EM_cluster(img, k, error = 10e-4, iter_n = 9999):
    #  init setting
    cnt = 0
    likelihood_arr = []
    # Initialise E-Step by KMeans
    means, cov, pis = kmeans_init(img, k)
    likelihood = 0
    new_likelihood = 2
    responsibilities = update_responsibility(img, means, cov, pis, k)
    while (abs(likelihood - new_likelihood) > error) and (cnt != iter_n):
        start_dt = datetime.datetime.now()
        cnt += 1
        likelihood = new_likelihood
        # M-Step
        labels = update_labels(responsibilities)
        # E-step
        responsibilities = update_responsibility(img, means, cov, pis, k)
        means = update_means(img, responsibilities)
        cov = update_covariance(img, responsibilities, means)
        pis = update_pis(responsibilities)
        new_likelihood = update_loglikelihood(img, means, cov, pis, k)
        likelihood_arr.append(new_likelihood)
        end_dt = datetime.datetime.now()
        diff = relativedelta(end_dt, start_dt)
        print("iter: %s, time interval: %s:%s:%s:%s" % (cnt, diff.hours, diff.minutes, diff.seconds, diff.microseconds))
    likelihood_arr = np.array(likelihood_arr)
    print('Converge at iteration {}'.format(cnt + 1))
    return labels, means, cov, pis, likelihood_arr



# FILENAME1 = './img/john-lawrence-sullivan-aw18.jpg'
FILENAME1 = 'D:\\Uni\\Spain\\MISA\\MISA\\P2_data\\1\\T1.nii'
FILENAME2 = 'D:\\Uni\\Spain\\MISA\\MISA\\P2_data\\1\\T2_FLAIR.nii'

orig_img_T1 = io.imread(FILENAME1)
orig_img_T1 = orig_img_T1[20,:,:]

orig_img_T2 = io.imread(FILENAME2)
orig_img_T2 = orig_img_T2[20,:,:]

orig_img = np.zeros([240,240,2])
orig_img[:,:,0] = orig_img_T1
orig_img[:,:,1] = orig_img_T2

x, y, z = orig_img.shape
plt.figure()
plt.axis("off")
plt.imshow(orig_img[:,:,1])
plt.title('Original Image T1');
plt.show()

img = flatten_img(orig_img)
labels, means, cov, pis, likelihood_arr = EM_cluster(img, 3)
em_img = recover_img(means[labels], X=x, Y=y, Z=z)
plt.figure()
plt.axis("off")
plt.imshow(em_img)
plt.title('Image Segmented by EM Algorithm');
plt.show()

    

