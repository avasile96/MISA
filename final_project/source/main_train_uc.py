# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 18:37:52 2021

@author: vasil
"""

# -*- coding: utf-8 -*-
"""Copy of Tutorial1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-LLta14B8NYhYfL5-zZeWBH-kxsa2yCy

**Tutorial # 1 - Introduction**

Goals:

1.   Train, validate, and test a segmentation network using keras
2.   Implement a multi-modality version and test its performance
3.   Test effect of hyperparameters (batch size, patch size, epochs, n kernels)
4.   Test effect of intensity standardisation
5.   Test effect of skip connections (segnet -> unet)
6.   Get used to keras

**Import libraries**
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt # plotting purposes

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import os

"""**Define parameters**"""

# dataset parameters
# FNAME_PATTERN = '../TrainingValidationTestSets/{}/{}/{}.nii.gz'
FNAME_PATTERN = '../TrainingValidationTestSets/{}/{}/{}.nii'
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
# MODEL_FNAME_PATTERN = './model.h5'
MODEL_FNAME_PATTERN = './model_bc.h5'
OPTIMISER = 'Adam'
LOSS = 'categorical_crossentropy'


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

"""**Load data**"""

def load_data(n_volumes=N_VOLUMES, image_size=IMAGE_SIZE, fname_pattern=FNAME_PATTERN, case = 'Training_Set') :
  volumes = np.zeros((n_volumes, *image_size))
  labels = np.zeros((n_volumes, *image_size, 1))
  counter = 0
  for i in os.listdir('../TrainingValidationTestSets/{}/'.format(case)):
    img_data = nib.load(fname_pattern.format(case,i,i+'_bc'))
    volumes[counter] = img_data.get_fdata()

    seg_data = nib.load(fname_pattern.format(case,i,i+'_seg'))
    labels[counter] = seg_data.get_fdata()
    counter += 1
  return (volumes, labels)


"""**Split into training, validation and testing**"""
(training_volumes, training_labels) = load_data(N_VOLUMES, IMAGE_SIZE, FNAME_PATTERN, case = 'Training_Set')
(validation_volumes, validation_labels) = load_data(N_VOLUMES, IMAGE_SIZE, FNAME_PATTERN, case = 'Validation_Set')
# (testing_volumes, testing_labels) = load_data(N_VOLUMES, IMAGE_SIZE, FNAME_PATTERN, case = 'Test_Set')

"""**Pre-process data**"""

def z_score_standardisation(x, avg, std):
  return (x-avg)/std

"""**Extract *useful* patches**

This step is fundamental, we want to provide the network with useful information
"""

def extract_patches(x, patch_size, patch_stride) :
  return tf.image.extract_patches(
    x,
    sizes=[1, *patch_size, 1],
    strides=[1, *patch_stride, 1],
    rates=[1, 1, 1, 1],
    padding='SAME', name=None)

def extract_useful_patches(
    volumes, labels,
    image_size=IMAGE_SIZE,
    patch_size=PATCH_SIZE,
    stride=PATCH_STRIDE,
    threshold=CONTENT_THRESHOLD,
    num_classes=N_CLASSES) :
  volumes = volumes.reshape([-1, image_size[1], image_size[2], 1])
  labels = labels.reshape([-1, image_size[1], image_size[2], 1])

  vol_patches = extract_patches(volumes, patch_size, stride).numpy()
  seg_patches = extract_patches(labels, patch_size, stride).numpy()

  vol_patches = vol_patches.reshape([-1, *patch_size, 1])
  seg_patches = seg_patches.reshape([-1, *patch_size, ])

  foreground_mask = seg_patches != 0

  useful_patches = foreground_mask.sum(axis=(1, 2)) > threshold * np.prod(patch_size)

  vol_patches = vol_patches[useful_patches]
  seg_patches = seg_patches[useful_patches]

  seg_patches = tf.keras.utils.to_categorical(
    seg_patches, num_classes=N_CLASSES, dtype='float32')
  
  return (vol_patches, seg_patches)


from scipy.ndimage import gaussian_filter

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, labels, shuffle=True):
        'Initialization'
        self.dim = PATCH_SIZE
        self.batch_size = BATCH_SIZE
        self.list_IDs = range(labels.shape[0])
        self.labels = labels
        self.n_channels = N_INPUT_CHANNELS
        self.n_classes = N_CLASSES
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=float)
        Y = np.empty((self.batch_size, *self.dim, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            map = self.labels[ID]

            sample = np.zeros_like(map)


            sample[map == 1] = np.random.normal(176.82642897, 62.63074249)
            sample[map == 2] = np.random.normal(232.12555423, 20.89609083)
            sample[map == 3] = np.random.normal(258.7664497, 24.62159618)

            sample = gaussian_filter(sample, sigma=np.abs(np.random.normal(1, 0.25)))

            noise = np.random.normal(0, 1, size=map.shape)

            sample = sample + noise

            sample[map == 0] = 0

            X[i, ] = (sample - 289.43550817663186) / 125.09902733275413

            # Store class
            Y[i, ] =  tf.keras.utils.to_categorical(map, num_classes=self.n_classes)

        return X, Y


# extract patches from training set
(training_patches, training_patches_seg) = extract_useful_patches(training_volumes, training_labels)

# extract patches from validation set
(validation_patches, validation_patches_seg) = extract_useful_patches(validation_volumes, validation_labels)


training_generator = DataGenerator(training_patches_seg)

for x in training_generator :
  break

plt.imshow(x[0][21, :, :, 0], cmap="gray")

plt.imshow(np.argmax(x[1][21, :, :, :], axis=-1))

"""Using callbacks to stop training and avoid overfitting


*   Early stopping with a certain patience
*   Save (and load!) best model


"""

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=PATIENCE),
    tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_FNAME_PATTERN, save_best_only=True)
]

# segnet = get_segnet()
# segnet.compile(optimizer=OPTIMISER, loss=LOSS)
# h = segnet.fit(
#     x=training_patches, 
#     y=training_patches_seg,
#     validation_data=(validation_patches, validation_patches_seg),
#     batch_size=BATCH_SIZE,
#     epochs=N_EPOCHS,
#     callbacks=my_callbacks,
#     verbose=1)

u_net = get_unet()
u_net.compile(optimizer=OPTIMISER, loss=LOSS)
h = u_net.fit(
    x=training_generator,
    # x=training_patches, 
    y=training_patches_seg,
    validation_data=(validation_patches, validation_patches_seg),
    batch_size=BATCH_SIZE,
    epochs=N_EPOCHS,
    callbacks=my_callbacks,
    verbose=1)


"""We could stop training when validation loss increases substantially"""

plt.figure()
plt.plot(range(N_EPOCHS), h.history['loss'], label='loss')
plt.plot(range(N_EPOCHS), h.history['val_loss'], label='val_loss')
plt.legend()
plt.show()