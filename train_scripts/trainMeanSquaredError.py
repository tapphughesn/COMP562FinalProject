from PIL import Image
from datetime import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from tensorflow import keras
import os

# Model Structure
model = keras.Sequential([
    keras.layers.Conv2D(4, 3, padding = 'same', activation='relu'),
    keras.layers.Conv2D(8, 3, padding = 'same', activation='relu'),
    keras.layers.MaxPool2D(pool_size = 2, padding = 'same'), 
    keras.layers.Conv2D(16, 3, padding = 'same', activation='relu'),
    keras.layers.Conv2D(16, 3, padding = 'same', activation='relu'),
    keras.layers.MaxPool2D(pool_size = 2, padding = 'same'), 
    keras.layers.Conv2D(32, 3, padding = 'same', activation='relu'),
    keras.layers.Conv2D(32, 3, padding = 'same', activation='relu'),
    keras.layers.MaxPool2D(pool_size = 2, padding = 'same'), 
    keras.layers.Conv2D(64, 3, padding = 'same', activation='relu'),
    keras.layers.Conv2D(64, 3, padding = 'same', activation='relu'),
    keras.layers.MaxPool2D(pool_size = 2, padding = 'same'), 
    keras.layers.Flatten(input_shape=(15,20,64)),
    keras.layers.Dense(4)
    ])

# Compile the model
model.compile(optimizer='adam', 
        loss = tf.keras.losses.MeanSquaredError(),
        metrics = ['accuracy'])


train_imgs_0 = []
train_labels_0 = []

train_file_names = glob('DATA/TRAIN/EOSINOPHIL/*') + glob('DATA/TRAIN/LYMPHOCYTE/*') + glob('DATA/TRAIN/MONOCYTE/*') + glob('DATA/TRAIN/NEUTROPHIL/*')
test_file_names = glob('DATA/TEST/EOSINOPHIL/*') + glob('DATA/TEST/LYMPHOCYTE/*') + glob('DATA/TEST/MONOCYTE/*') + glob('DATA/TEST/NEUTROPHIL/*')

# Make a dictionary mapping the first letter of the cell type to the label integer
class_dict = {}
class_dict['E'] = 0
class_dict['L'] = 1
class_dict['M'] = 2
class_dict['N'] = 3

# Shuffle file names to mix up order of training
np.random.shuffle(train_file_names)
np.random.shuffle(test_file_names)

# Make train_imgs and train_labels for this section
train_imgs = []
train_labels = []
for f in train_file_names:
    im = np.asarray(Image.open(f)).astype(np.float16)
    train_imgs.append(im)
    # 11th letter in file name is the first letter of the cell type
    label = class_dict[f[11]]
    train_labels.append(label)

train_imgs = np.asarray(train_imgs)
train_labels = np.asarray(train_labels).astype(np.float16)

# Make train_imgs and train_labels for this section
test_imgs = []
test_labels = []
for f in test_file_names:
    im = np.asarray(Image.open(f)).astype(np.float16)
    test_imgs.append(im)
    # 10th letter in test file name is the first letter of the cell type
    label = class_dict[f[10]]
    test_labels.append(label)

test_imgs = np.asarray(test_imgs)
test_labels = np.asarray(test_labels).astype(np.float16)

# Create checkpoint to save weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/mean_squared_error.cpkt",
        save_weights_only = True,
        verbose = 1)

# use Model.fit() to train the model
model.fit(train_imgs, train_labels, validation_data=(test_imgs, test_labels), epochs=100, verbose=1, callbacks=[cp_callback], shuffle=False)

# print model summary for information abt the model
print(model.summary())
     
# Save Model weights
# model.save_weights('checkpoints/' + datetime.now().strftime("%m-%d-%Y-%H:%M:%S") + "_parameters")
     
