# -*- coding: utf-8 -*-
"""Tutorial3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1555EZvmwwCRaISPdHCe4Suv8nK_0VXBO

**Tutorial # 3 - Contrast agnostic training**

Goals:

1.   Create synthetic images
2.   Determine GMM parameters

**Import libraries**
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt # plotting purposes

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

"""**Define parameters**"""

# dataset parameters
FNAME_PATTERN = 'drive/MyDrive/iSeg2019-Training/subject-{}-{}.hdr'
N_VOLUMES = 10
IMAGE_SIZE = (144, 192, 256)

# network parameters
N_CLASSES = 4
N_INPUT_CHANNELS = 1
PATCH_SIZE = (32, 32)
PATCH_STRIDE = (32, 32)

# training, validation, test parameters
TRAINING_VOLUMES = [0, 1, 2, 3, 4, 5, 6]
VALIDATION_VOLUMES = [7, 8]
TEST_VOLUMES = [9]

# data preparation parameters
CONTENT_THRESHOLD = 0.3

# training parameters
N_EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 20
MODEL_FNAME_PATTERN = 'model.h5'
OPTIMISER = 'Adam'
LOSS = 'categorical_crossentropy'

"""**Mount drive**"""

from google.colab import drive
drive.mount("/content/drive")

"""**Define SegNet architecture**"""

def get_unet(img_size=PATCH_SIZE, n_classes=N_CLASSES, n_input_channels=N_INPUT_CHANNELS, scale=1):
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
    concat1 = layers.concatenate([conv3, up1], axis=-1)
    conv4 = layers.Conv2D(128*scale, (3, 3), padding="same", activation='relu')(concat1)
    
    up2 = layers.UpSampling2D((2, 2))(conv4)
    concat2 = layers.concatenate([conv2, up2], axis=-1)
    conv5 = layers.Conv2D(64*scale, (3, 3), padding="same", activation='relu')(concat2)
    
    up3 = layers.UpSampling2D((2, 2))(conv5)
    concat3 = layers.concatenate([conv1, up3], axis=-1)

    conv6 = layers.Conv2D(32*scale, (3, 3), padding="same", activation='relu')(concat3)

    outputs = layers.Conv2D(n_classes, (1, 1), activation="softmax")(conv6)

    model = keras.Model(inputs, outputs)

    return model

"""**Load data**"""

def load_data(n_volumes=N_VOLUMES, image_size=IMAGE_SIZE, fname_pattern=FNAME_PATTERN) :
  T1_volumes = np.zeros((n_volumes, *image_size, 1))
  T2_volumes = np.zeros((n_volumes, *image_size, 1))
  labels = np.zeros((n_volumes, *image_size, 1))
  for i in range(n_volumes) :
    img_data = nib.load(fname_pattern.format(i+1, 'T1'))
    T1_volumes[i] = img_data.get_fdata()

    img_data = nib.load(fname_pattern.format(i+1, 'T2'))
    T2_volumes[i] = img_data.get_fdata()

    seg_data = nib.load(fname_pattern.format(i+1, 'label'))
    labels[i] = seg_data.get_fdata()

  return (T1_volumes, T2_volumes, labels)

(T1_volumes, _, labels) = load_data()

"""**Split into training, validation and testing**"""

training_volumes_T1 = T1_volumes[TRAINING_VOLUMES]
training_labels = labels[TRAINING_VOLUMES]

validation_volumes_T1 = T1_volumes[VALIDATION_VOLUMES]
validation_labels = labels[VALIDATION_VOLUMES]

"""**Determine GMM parameters**"""

import sklearn.mixture

training_volume_T1=training_volumes_T1[0]

A=sklearn.mixture.GaussianMixture(3).fit(training_volume_T1[training_labels[0]!=0].reshape([-1, 1]))

print(A.means_.flatten(), A.covariances_.flatten())

plt.hist(training_volume_T1[training_labels[0]!=0], 100);

"""**Pre-process data**"""

def z_score_standardisation(x, avg, std):
  return (x-avg)/std

ref_avg = training_volumes_T1[training_labels!=0].mean()
ref_std = training_volumes_T1[training_labels!=0].std()

training_volumes_T1 = z_score_standardisation(training_volumes_T1, ref_avg, ref_std)
validation_volumes_T1 = z_score_standardisation(validation_volumes_T1, ref_avg, ref_std)

ref_avg, ref_std

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

# extract patches from training set
(training_patches_seg, _) = extract_useful_patches(training_labels, training_labels)

# extract patches from validation set
(validation_patches_T1, validation_patches_seg) = extract_useful_patches(validation_volumes_T1, validation_labels)

import numpy as np
import random
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

training_generator = DataGenerator(training_patches_seg)

for x in training_generator :
  break

plt.imshow(x[0][21, :, :, 0], cmap="gray")

plt.imshow(np.argmax(x[1][21, :, :, :], axis=-1))

"""**Train network**"""

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=PATIENCE),
    tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_FNAME_PATTERN)
]

# Generators
training_generator = DataGenerator(training_patches_seg)

segnet = get_unet()
segnet.compile(optimizer=keras.optimizers.SGD(learning_rate=1e-3), loss=LOSS)
segnet.fit(
    x=training_generator,
    validation_data=(validation_patches_T1, validation_patches_seg),
    epochs=N_EPOCHS,
    callbacks=my_callbacks,
    verbose=1)

"""**Load best model**"""

segnet = get_unet(
    img_size=(IMAGE_SIZE[1], IMAGE_SIZE[2]),
    n_classes=N_CLASSES,
    n_input_channels=N_INPUT_CHANNELS)
segnet.compile(optimizer=OPTIMISER, loss=LOSS)
segnet.load_weights('model.h5')

"""**Prepare test data**"""

testing_volumes_T1 = T1_volumes[TEST_VOLUMES]
testing_labels = labels[TEST_VOLUMES]
testing_volumes_T1 = z_score_standardisation(testing_volumes_T1, ref_avg, ref_std)

testing_volumes_T1_processed = testing_volumes_T1.reshape([-1, IMAGE_SIZE[1], IMAGE_SIZE[2], 1])
testing_labels_processed = testing_labels.reshape([-1, IMAGE_SIZE[1], IMAGE_SIZE[2], 1])

testing_labels_processed = tf.keras.utils.to_categorical(
    testing_labels_processed, num_classes=4, dtype='float32')

"""**Predict labels for test data**"""

prediction = segnet.predict(x=testing_volumes_T1_processed)

prediction = np.argmax(prediction, axis=3)

plt.imshow(prediction[:, :, 150])

"""**Compute DSC for test data**"""

def compute_dice(prediction, labels) :
  prediction = prediction.squeeze()
  labels = labels.squeeze()
  for c in np.unique(prediction) :
    intersection = np.logical_and(prediction == c, labels==c).sum()
    union = (prediction == c).sum() + (labels==c).sum()
    print(f'Dice coefficient class {c} equal to {2 * intersection / union : .2f}')

compute_dice(prediction, testing_labels)