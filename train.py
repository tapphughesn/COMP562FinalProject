from PIL import Image
import datetime
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
# There are approximately 10,000 images in the training dataset
# Train on sections of the dataset so it can fit in my GPU memory (8GB)

train_imgs_0 = []
train_labels_0 = []

for f in glob('DATA/TRAIN/EOSINOPHIL/*'):
    im = np.asarray(Image.open(f)).astype(np.float16)
    train_imgs_0.append(im)
    train_labels_0.append(0)

for f in glob('DATA/TRAIN/LYMPHOCYTE/*'):
    im = np.asarray(Image.open(f)).astype(np.float16)
    train_imgs_0.append(im)
    train_labels_0.append(1)

for f in glob('DATA/TRAIN/MONOCYTE/*'):
    im = np.asarray(Image.open(f)).astype(np.float16)
    train_imgs_0.append(im)
    train_labels_0.append(2)

for f in glob('DATA/TRAIN/NEUTROPHIL/*'):
    im = np.asarray(Image.open(f)).astype(np.float16)
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
train_labels = np.asarray(train_labels).astype(np.float16)
print("Imgs size: ", sys.getsizeof(train_imgs))
print("Labels size: ", sys.getsizeof(train_labels))

print(tf.shape(train_imgs))
print(tf.shape(train_labels))

# use Model.fit() to train the model
batch_size = 100
num_batches = len(train_imgs) // batch_size
for batch in range(len(num_batches)):
    if (((batch+1)*batch_size) > len(train_imgs)):
        break
    model.fit(train_imgs[batch], train_labels[batch], batch_size=4, epochs=1, verbose=1, callbacks=[model_checkpoint], shuffle=False)

# Save Model weights
model.save_weights('checkpoints/' + datetime.now().strftime("%m-%d-%Y-%H:%M:%S") + "_parameters")

