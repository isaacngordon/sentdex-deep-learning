# !/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard 
import pickle
import time

NAME = "Cats-Dogs-CNN-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# GPU Options
# gpu_options = tf.GPUOptions(per_proccess_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# reload pickle saved data
x = pickle.load(open('x-sentdex-2.pickle', 'rb'))
y = pickle.load(open('y-sentdex-2.pickle', 'rb'))

# normalize the data, we know pixel data is 0-255 so we will use that simply
x = x/255.0

# create model (this is 5 layers)
# input layer
model = Sequential()

# first Conv layer
model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# second conv layer
model.add(Conv2D(64, (3, 3), input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# dense last hidden layer
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

# output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

# compile
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# run, set batch size to something responsable subjective to data set size
model.fit(x,
          y,
          batch_size=32,
          epochs=10,
          validation_split=0.33,
          callbacks=[tensorboard])


