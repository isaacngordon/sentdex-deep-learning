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

START_TIME = int(time.time())
print('Start Time: {}'.format(START_TIME))

# GPU Options
# gpu_options = tf.GPUOptions(per_proccess_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# reload pickle saved data
x = pickle.load(open('x-sentdex-2.pickle', 'rb'))
y = pickle.load(open('y-sentdex-2.pickle', 'rb'))

# normalize the data, we know pixel data is 0-255 so we will use that simply
x = x/255.0

# model variation testing variables
dense_layers = [0, 1, 2]
conv_layers = [1, 2, 3]
layer_sizes = [32, 64, 128]

# iterate through model variations and run them each
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            # create log 
            LOG_NAME = 'sentdex5-CNN_{}-conv_{}-dense_{}-nodes_{}'.format(conv_layer, dense_layer, layer_size, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(LOG_NAME))

            # create model (this is 5 layers)
            # input layer
            model = Sequential()

            # first Conv layer
            model.add(Conv2D(layer_size, (3, 3), input_shape=x.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # subsequent conv layers
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3), input_shape=x.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Flatten())
            
            # generate dense layers
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            # output layer
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            # compile
            model.compile(loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])

            # run, set batch size to something responsable subjective to data set size
            print('Running {} now'.format(LOG_NAME))
            model.fit(x,
                    y,
                    batch_size=32,
                    epochs=10,
                    validation_split=0.33,
                    callbacks=[tensorboard])

END_TIME = int(time.time())
print('End Time: {}'.format(END_TIME))
print('Total Run Time: {}'.format(START_TIME - END_TIME))