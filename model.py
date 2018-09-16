import os
import csv
import cv2
import numpy as np
import keras
from scipy import ndimage
from random import shuffle


# read in udacity data from file
lines=[]
with open('../data_provided_by_udacity/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    i_have_seen_firstline=False
    for line in reader:
        if i_have_seen_firstline:
            lines.append(line)
        else:
            i_have_seen_firstline = True

import sklearn

# split them into a training and a validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)  

#define generator

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #name = './IMG/'+batch_sample[0].split('/')[-1]
                current_path = '../data_provided_by_udacity/IMG/' + batch_sample[0].split('/')[-1] 
                current_left_path = '../data_provided_by_udacity/IMG/' + batch_sample[1].split('/')[-1] 
                current_right_path = '../data_provided_by_udacity/IMG/' + batch_sample[2].split('/')[-1] 
                #center_image = cv2.imread(current_path)
                center_image = ndimage.imread(current_path)
                left_image = ndimage.imread(current_left_path) 
                right_image = ndimage.imread(current_right_path)
                #center_image = cv2.cvtColor(ndimage.imread(current_path), cv2.COLOR_RBG2YUV)
                #left_image = cv2.cvtColor(ndimage.imread(current_left_path) , cv2.COLOR_RBG2YUV)
                #right_image = cv2.cvtColor(ndimage.imread(current_right_path), cv2.COLOR_RBG2YUV)
                center_angle = float(batch_sample[3])
                correction = 0.003  # this is a parameter to tune 0.03 was not bad
                left_angle = center_angle + correction
                right_angle = center_angle - correction
                #left_angle = center_angle *1.15
                #ight_angle = center_angle - 1.15
                #optionally use left and right cameras
                use_all_cameras = True
                if use_all_cameras:
                    images.extend([center_image, left_image,right_image])
                    angles.extend([center_angle,left_angle,right_angle])
                else:
                    images.append(center_image)
                    angles.extend(center_angle)
            #optionally augment by flipping all images right curves <> left curves    
            augment_by_flipping=True
            if augment_by_flipping:
                augmented_images, augmented_angles = [],[]
                for image,angle in zip(images, angles):
                    augmented_images.append(image)
                    augmented_angles.append(angle)
                    #augmented_images.append(cv2.flip(image,1))
                    augmented_images.append(np.fliplr(image))
                    augmented_angles.append(angle*-1.0)
            else:
                augmented_images, augmented_angles =images,angles

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt

# compile and train the model using the generator function
my_batch_size= 16 #128
train_generator = generator(train_samples, batch_size=my_batch_size)
validation_generator = generator(validation_samples, batch_size=my_batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

# optionally perform dropout in some layers, see below
dropout_prob=0.0#0.8

model=Sequential()
#model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
#normalize data
model.add(Lambda(lambda x: x/127.5 - 1.,  #
        input_shape=(row, col,ch))) #,
        #output_shape=(row, col, ch)))

#optionally apply cropping
cropping= True
if cropping:
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

model.add(Dropout(dropout_prob))
    
##### 1st convolutional layer:
model.add(Conv2D(24, kernel_size=(5, 5),
                 strides = (2,2),
                 activation='relu',
                 padding='valid'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(dropout_prob))

##### 2nd convolutional layer:
model.add(Conv2D(36, kernel_size=(5, 5),
                 strides = (2,2),
                 activation='relu',
                 padding='valid'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(dropout_prob))

##### 3rd convolutional layer:
model.add(Conv2D(48, kernel_size=(5, 5),
                 strides = (2,2),
                 activation='relu',
                 padding='valid'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(dropout_prob))

##### 4th convolutional layer:
model.add(Conv2D(64, kernel_size=(3, 3),
                 strides = (1,1),
                 activation='relu',
                 padding='valid'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(dropout_prob))

##### 5th convolutional layer:
model.add(Conv2D(64, kernel_size=(3, 3),
                 strides = (1,1),
                 activation='relu',
                 padding='valid'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(dropout_prob))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(dropout_prob))

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(dropout_prob))

model.add(Dense(10))
model.add(Activation('relu'))
#model.add(Dropout(dropout_prob))

model.add(Dense(1))


#model.summary()

model.compile(loss='mse',optimizer='adam')

history_object =  model.fit_generator(train_generator, steps_per_epoch= len(train_samples)/my_batch_size, 
                    epochs=4,  verbose=1,
            validation_data=validation_generator, validation_steps= len(validation_samples)/my_batch_size, use_multiprocessing=True
            )

# save the model
model.save('model.h5')


#plot validation and training losses over time
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()


##############
