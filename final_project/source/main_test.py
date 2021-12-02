# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 20:33:18 2021

@author: vasil
"""

"""**Load best model**"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt # plotting purposes
import cv2

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import os
import gc

"""**Define parameters**"""

# dataset parameters
FNAME_PATTERN = '../TrainingValidationTestSets/{}/{}/{}.nii.gz'
N_VOLUMES = 10
IMAGE_SIZE = (256, 128, 256)

# network parameters
N_CLASSES = 4
N_INPUT_CHANNELS = 1
PATCH_SIZE = (32, 32)
PATCH_STRIDE = (32, 32)

# data preparation parameters
CONTENT_THRESHOLD = 0.3

# training parameters
N_EPOCHS = 10
BATCH_SIZE = 32
PATIENCE = 10
MODEL_FNAME_PATTERN = 'model.h5'
OPTIMISER = 'Adam'
LOSS = 'categorical_crossentropy'

"""**Load data**"""

def load_data(n_volumes=N_VOLUMES, image_size=IMAGE_SIZE, fname_pattern=FNAME_PATTERN, case = 'Training_Set') :
  volumes = np.zeros((n_volumes, *image_size, 1))
  labels = np.zeros((n_volumes, *image_size, 1))
  counter = 0
  for i in os.listdir('../TrainingValidationTestSets/{}/'.format(case)):
    img_data = nib.load(fname_pattern.format(case,i,i))
    volumes[counter] = img_data.get_fdata()

    seg_data = nib.load(fname_pattern.format(case,i,i+'_seg'))
    labels[counter] = seg_data.get_fdata()
    counter += 1
  return (volumes, labels)

"""**Split into training, validation and testing**"""
(testing_volumes, testing_labels) = load_data(N_VOLUMES, IMAGE_SIZE, FNAME_PATTERN, case = 'Validation_Set')


"""**Define SegNet architecture**"""

def get_segnet(img_size=PATCH_SIZE, n_classes=N_CLASSES, n_input_channels=N_INPUT_CHANNELS, scale=1):
    inputs = keras.Input(shape=img_size + (n_input_channels, ))

    # Encoding path
    conv1 = layers.Conv2D(32*scale, (3, 3), padding="same", activation='relu')(inputs)
    max1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(64*scale, (3, 3), padding="same", activation='relu')(max1)
    max2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(128*scale, (3, 3), padding="same", activation='relu')(max2)
    max3 = layers.MaxPooling2D((2, 2))(conv3)

    lat = layers.Conv2D(256*scale, (3, 3), padding="same", activation='relu')(max3)

    # Decoding path
    up1 = layers.UpSampling2D((2, 2))(lat)
    conv4 = layers.Conv2D(128*scale, (3, 3), padding="same", activation='relu')(up1)
    
    up2 = layers.UpSampling2D((2, 2))(conv4)
    conv5 = layers.Conv2D(64*scale, (3, 3), padding="same", activation='relu')(up2)
    
    up3 = layers.UpSampling2D((2, 2))(conv5)
    conv6 = layers.Conv2D(32*scale, (3, 3), padding="same", activation='relu')(up3)

    outputs = layers.Conv2D(n_classes, (1, 1), activation="softmax")(conv6)

    model = keras.Model(inputs, outputs)

    return model

def get_unet(img_size=PATCH_SIZE, n_classes=N_CLASSES, n_input_channels=N_INPUT_CHANNELS, scale=1):
    inputs = keras.Input(shape=img_size + (n_input_channels, ))

    # Encoding path
    conv1 = layers.Conv2D(32*scale, (3, 3), padding="same", activation='relu')(inputs)
    drop1 = layers.Dropout(rate=0.2)(conv1, training=True)
    max1 = layers.MaxPooling2D((2, 2))(drop1)

    conv2 = layers.Conv2D(64*scale, (3, 3), padding="same", activation='relu')(max1)
    drop2 = layers.Dropout(rate=0.2)(conv2, training=True)
    max2 = layers.MaxPooling2D((2, 2))(drop2)

    conv3 = layers.Conv2D(128*scale, (3, 3), padding="same", activation='relu')(max2)
    drop3 = layers.Dropout(rate=0.2)(conv3, training=True)
    max3 = layers.MaxPooling2D((2, 2))(drop3)

    lat = layers.Conv2D(256*scale, (3, 3), padding="same", activation='relu')(max3)
    drop4 = layers.Dropout(rate=0.2)(lat, training=True)

    # Decoding path
    up1 = layers.UpSampling2D((2, 2))(drop4)
    concat1 = layers.concatenate([conv3, up1], axis=-1)
    conv4 = layers.Conv2D(128*scale, (3, 3), padding="same", activation='relu')(concat1)
    drop5 = layers.Dropout(rate=0.2)(conv4, training=True)
    
    up2 = layers.UpSampling2D((2, 2))(drop5)
    concat2 = layers.concatenate([conv2, up2], axis=-1)
    conv5 = layers.Conv2D(64*scale, (3, 3), padding="same", activation='relu')(concat2)
    drop6 = layers.Dropout(rate=0.2)(conv5, training=True)
    
    up3 = layers.UpSampling2D((2, 2))(drop6)
    concat3 = layers.concatenate([conv1, up3], axis=-1)

    conv6 = layers.Conv2D(32*scale, (3, 3), padding="same", activation='relu')(concat3)
    drop7 = layers.Dropout(rate=0.2)(conv6, training=True)

    outputs = layers.Conv2D(n_classes, (1, 1), activation="softmax")(drop7)

    model = keras.Model(inputs, outputs)

    return model

# segnet = get_segnet(
#     img_size=(IMAGE_SIZE[1], IMAGE_SIZE[2]),
#     n_classes=N_CLASSES,
#     n_input_channels=N_INPUT_CHANNELS)
# segnet.compile(optimizer=OPTIMISER, loss=LOSS)
# segnet.load_weights('model.h5')

u_net = get_unet(
    img_size=(IMAGE_SIZE[1], IMAGE_SIZE[2]),
    n_classes=N_CLASSES,
    n_input_channels=N_INPUT_CHANNELS)
u_net.compile(optimizer=OPTIMISER, loss=LOSS)
u_net.load_weights('unet.h5')



"""**Prepare test data**"""

testing_volumes_processed = testing_volumes.reshape([-1, IMAGE_SIZE[1], IMAGE_SIZE[2], 1])
testing_labels_processed = testing_labels.reshape([-1, IMAGE_SIZE[1], IMAGE_SIZE[2], 1])

testing_labels_processed_cat = tf.keras.utils.to_categorical(
    testing_labels_processed, num_classes=4, dtype='float32')

# """**Predict labels for test data**"""

# prediction = segnet.predict(x=testing_volumes_processed)
prediction = u_net.predict(x=testing_volumes_processed)


del testing_volumes
gc.collect()


prediction = np.argmax(prediction, axis=3)
testing_labels_processed_cat = np.argmax(testing_labels_processed_cat, axis=3)


"""**Compute DSC for test data**"""

def compute_dice(prediction, labels) :
  prediction = prediction.squeeze()
  labels = labels.squeeze()
  
  
  for c in np.unique(prediction) :
    intersection = np.logical_and(prediction == c, labels == c).sum()
    union = (prediction == c).sum() + (labels==c).sum()
    print(f'Dice coefficient class {c} equal to {2 * intersection / union : .2f}')

compute_dice(prediction, testing_labels_processed_cat)

cv2.imwrite('damnbro.png', np.array(prediction[150, :, :],dtype = np.uint8))
cv2.imwrite('labels.png', np.array(testing_labels_processed_cat[150, :, :],dtype = np.uint8))

N_RUNS = 10
# predictions = np.zeros(IMAGE_SIZE+(N_CLASSES, N_RUNS, ))
predictions = []
for run in range(N_RUNS) :
  predictions.append(u_net.predict(x=testing_volumes_processed))

mean_prediction = np.array(predictions).mean(axis=-1)
std_prediction = np.array(predictions).std(axis=-1)


cv2.imwrite('damnbro.png', np.array(prediction[150, :, :],dtype = np.uint8))
# plt.imshow(std_prediction[:, :, 150, 2])



