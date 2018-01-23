#!/usr/bin/python
import keras
import cv2
import numpy as np
import os
import math

#------------------------------------------------------------------------------
# read all dataset for training
#------------------------------------------------------------------------------
traindir = 'data/FER2013Train/'
d_ = {}
maxnum = 0
for i in range(8):
    fname = 'lists/train-ori-c%d.txt' % (i)
    f = open(fname)
    alllines = f.readlines()
    f.close()
    if maxnum < len(alllines):
        maxnum = len(alllines)
    ##
    d_[i] = alllines
##
##
x_train = np.zeros((maxnum*8, 64, 64, 1), dtype='f')
x_p = np.zeros((maxnum*8, 64, 64, 1), dtype='f')
y_p = np.zeros((maxnum*8), dtype=np.uint8)

idxout = 0
for i in range(8):
    data = d_[i]
    num = 0
    idx = 0    
    while num < maxnum:
        sp = (data[idx]).split(" ")
        y_p[idxout ] = int(sp[1])
        img = cv2.imread(os.path.join(traindir, sp[0]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64,64))
        x_p[idxout, :, :, 0] = img
        ###
        idxout = idxout + 1
        idx = idx + 1
        if idx >= len(data):
            idx = 0
        num = num + 1
    ###
###

randomid = np.random.permutation(x_p.shape[0])
y_p = keras.utils.to_categorical(y_p, num_classes=8)
y_train = np.zeros(y_p.shape, dtype=np.uint8)
for i in range(x_p.shape[0]):
    x_train[i, :, :, :] = x_p[randomid[i], :, :, :].copy()
    y_train[i] = y_p[randomid[i]]
x_p = 0
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# read all dataset for validation
#------------------------------------------------------------------------------
valdir = 'data/FER2013Valid/'
f = open('lists/validation.txt')
alllines = f.readlines()
f.close()

x_val = np.zeros((len(alllines), 64, 64, 1), dtype='f')
y_p = np.zeros((len(alllines)), dtype=np.uint8)

idx = 0
for line in alllines:
    sp = line.split(" ")
    y_p[idx] = int(sp[1])
    img = cv2.imread(os.path.join(valdir, sp[0]), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64,64))
    x_val[idx, :, :, 0] = img
    idx = idx + 1
###

y_val = keras.utils.to_categorical(y_p, num_classes=8)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Model defenition
#------------------------------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator

### data augmentation
datagen = ImageDataGenerator(featurewise_center=True, 
                             featurewise_std_normalization=True, \
                             width_shift_range=0.08, \
                             height_shift_range=0.08, \
                             zoom_range=0.05, \
                             rotation_range=20, \
                             shear_range=0.05, \
                             horizontal_flip=True)

### model description
vgg13 = keras.models.Sequential()

vgg13.add(keras.layers.Conv2D(64, (3, 3), activation='relu', 
                              input_shape=(64,64,1), padding='same'))
vgg13.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
vgg13.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
vgg13.add(keras.layers.Dropout(0.25))

vgg13.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
vgg13.add(keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
vgg13.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
vgg13.add(keras.layers.Dropout(0.25))

vgg13.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
vgg13.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
vgg13.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
vgg13.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
vgg13.add(keras.layers.Dropout(0.25))

vgg13.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
vgg13.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
vgg13.add(keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
vgg13.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
vgg13.add(keras.layers.Dropout(0.25))

vgg13.add(keras.layers.Flatten())
vgg13.add(keras.layers.Dense(1024, activation='relu'))
vgg13.add(keras.layers.Dropout(0.5))

vgg13.add(keras.layers.Dense(1024, activation='relu'))
vgg13.add(keras.layers.Dropout(0.5))

vgg13.add(keras.layers.Dense(8, activation='softmax'))
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Training Process
#------------------------------------------------------------------------------

def steplr(epoch):
    lr = 0.025
    max_epochs=100.0
    lr = lr * (1.0 - epoch/max_epochs)
    return lr

sgd = keras.optimizers.SGD(lr=0.025, decay=0.0005, momentum=0.9, nesterov=True)
vgg13.compile(loss='categorical_crossentropy', optimizer=sgd,  
              metrics=['accuracy'])

datagen.fit(x_train)

vgg13.fit_generator(datagen.flow(x_train, y_train, batch_size=128), 
                    steps_per_epoch=x_train.shape[0]/128, 
                    epochs=100, 
                    validation_data=datagen.flow(x_val, y_val, batch_size=128),
                    validation_steps=x_val.shape[0]/128,
                    callbacks=[
                        keras.callbacks.LearningRateScheduler(steplr, verbose=1),
                        keras.callbacks.ModelCheckpoint('vgg13-baseline.h5', 
                            monitor='val_acc', 
                            verbose=1,
                            save_best_only=True)
                        ]
                    )
#------------------------------------------------------------------------------



