from PIL import Image
import sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import tensorflow as tf
from tensorflow import keras


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
    keras.layers.Dense(128, input_shape=np.asarray([30,40,3]), activation='relu'),
    keras.layers.Dense(3, input_shape=np.asarray([30,40,3]), activation='softmax')
    ])

# Compile the model
model.compile(optimizer='adam', 
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ['accuracy'])

# Create Model callback to save checkpoint with the best weights
model_checkpoint = keras.callbacks.ModelCheckpoint('checkpoints', monitor="loss", verbose=0)

# train model on sections of the dataset

    train_imgs_0 = []
    train_labels_0 = []
    
    for f in glob('DATA/TRAIN/EOSINOPHIL/*'):
        im = np.asarray(Image.open(f))
        train_imgs_0.append(im)
        train_labels_0.append(0)
    
    for f in glob('DATA/TRAIN/LYMPHOCYTE/*'):
        im = np.asarray(Image.open(f))
        train_imgs_0.append(im)
        train_labels_0.append(1)
    
    for f in glob('DATA/TRAIN/MONOCYTE/*'):
        im = np.asarray(Image.open(f))
        train_imgs_0.append(im)
        train_labels_0.append(2)
    
    for f in glob('DATA/TRAIN/NEUTROPHIL/*'):
        im = np.asarray(Image.open(f))
        train_imgs_0.append(im)
        train_labels_0.append(3)
    
    # Shuffle the dataset and labels with the same shuffle
    permutation = np.arange(len(train_imgs_0))
    np.random.shuffle(permutation)
    train_imgs = []
    train_labels = []
    for i in range(len(train_imgs_0)):
        index = permutation[i]
        train_imgs.append(train_imgs_0[index])
        train_labels.append(train_labels_0[index])
    
    del train_imgs_0
    del train_labels_0
    train_imgs = np.asarray(train_imgs).astype(np.float16)
    train_labels = np.asarray(train_labels).astype(np.float16)
    print("Imgs size: ", sys.getsizeof(train_imgs))
    print("Labels size: ", sys.getsizeof(train_labels))
    train_imgs = tf.convert_to_tensor(train_imgs, dtype=tf.float16)
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float16)
    
    
    # use Model.fit() to train the model
    model.fit(train_imgs, train_labels, batch_size=4, epochs=1, verbose=1, callbacks=[model_checkpoint], shuffle=False)
