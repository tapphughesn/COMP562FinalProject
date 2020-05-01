from PIL import Image
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from tensorflow import keras
import os

#Stop tensorflow from seeing my gpu bc I can't get it to use my gpu correctly anyway
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Model Structure
model = keras.Sequential([
    keras.layers.Conv2D(8, 3, padding = 'same', activation='relu'),
    keras.layers.Conv2D(16, 3, padding = 'same', activation='relu'),
    keras.layers.MaxPool2D(pool_size = 2, padding = 'same'), 
    keras.layers.Conv2D(32, 3, padding = 'same', activation='relu'),
    keras.layers.Conv2D(64, 3, padding = 'same', activation='relu'),
    keras.layers.MaxPool2D(pool_size = 2, padding = 'same'), 
    keras.layers.Conv2D(128, 3, padding = 'same', activation='relu'),
    keras.layers.Conv2D(256, 3, padding = 'same', activation='relu'),
    keras.layers.MaxPool2D(pool_size = 2, padding = 'same'), 
    keras.layers.Flatten(input_shape=(30,40,3)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(4)
    ])

# Compile the model
model.compile(optimizer='adam', 
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics = ['accuracy'])

# train model on sections of the dataset
# There are approximately 10,000 images in the training dataset
# For some reason data can't exceed 10% of my system memory (16GB -> 1.6GB)

train_imgs_0 = []
train_labels_0 = []

file_names = glob('DATA/TRAIN/EOSINOPHIL/*') + glob('DATA/TRAIN/LYMPHOCYTE/*') + glob('DATA/TRAIN/MONOCYTE/*') + glob('DATA/TRAIN/NEUTROPHIL/*')

# Make a dictionary mapping the first letter of the cell type to the label integer
class_dict = {}
class_dict['E'] = 0
class_dict['L'] = 1
class_dict['M'] = 2
class_dict['N'] = 3

# Shuffle file names to mix up order of training
np.random.shuffle(file_names)

# Make train_imgs and train_labels for this section
train_imgs = []
train_labels = []
for f in file_names:
    im = np.asarray(Image.open(f)).astype(np.float16)
    train_imgs.append(im)
    # 11th letter in file name is the first letter of the cell type
    label = class_dict[f[11]]
    train_labels.append(3)

train_imgs = np.asarray(train_imgs)
train_labels = np.asarray(train_labels).astype(np.float16)

# use Model.fit() to train the model
model.fit(train_imgs, train_labels, epochs=1, verbose=1, shuffle=False)
     
# Save Model weights
model.save_weights('checkpoints/' + datetime.now().strftime("%m-%d-%Y-%H:%M:%S") + "_parameters")
     
