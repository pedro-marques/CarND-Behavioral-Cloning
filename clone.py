import csv
import cv2
import numpy as np

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    # Read the center, left and right images paths respectively
    center_image_source_path = line[0]
    left_image_source_path = line[1]
    right_image_source_path = line[2]
    #print(source_path)

    # TODO: Review this part of AWS instance
    # If using an AWS instance
    #filename = source_path.split('/')[-1]
    #current_path = '/data/IMG/' + source_path
    #filepath='data/'
    #image = cv2.imread(filepath+source_path) # If using an AWS instance change source_path to current_path
    image = cv2.imread(center_image_source_path)
    # Great tip from Andreas, Arnaldo Gunzi - cv2.imread reads the image in BGR, but the code in drive.py
    # reads in RGB so we convert it
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Left image path
    image = cv2.imread(left_image_source_path)
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Right image path
    image = cv2.imread(right_image_source_path)
    images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Read the steering angle
    measurement = float(line[3])
    measurements.append(measurement)

    # Center the steering angles for the left and right images respectively
    # by adding and subtracting a correction value to the original steering angle
    correction_value = 0.2

    # For the left side image
    measurements.append(measurement+correction_value)

    # For the right side image
    measurements.append(measurement-correction_value)

    # NOTE: The steering angle measurements must be added to the array 'measurements' in that specific order
    # because of the way the data is saved in the driving_log.csv file: [center_image][left_image][right_image]...

# Data Preprocessing
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(np.fliplr(image))
    augmented_measurements.append(measurement*-1)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print('Image shape:',X_train.shape)
print('Labels shape:',y_train.shape)

## Network Architecture
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
#import tensorflow as tf
from keras.backend import tf as ktf

model = Sequential()

# Crop 50 pixels from the top of the image (mainly trees and the sky),
# 20 pixels from the bottom (hood of the car), 0 from the left and right
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

# Normalization Layer
#model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 127.5) - 1, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))

# Resize the images
#model.add(Lambda(lambda image: ktf.image.resize_images(image, (64,64))))

# NVIDIA Architecture
# 3 5x5 Convolutional Layers
model.add(Convolution2D(48, 3, 3, border_mode='valid'))
model.add(Activation('elu'))
model.add(MaxPooling2D((2, 2), border_mode='valid'))

model.add(Convolution2D(24, 5, 5, border_mode='valid'))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), border_mode='valid'))

# 2 3x3 Convolutional Layers

# Flatten Layer
model.add(Flatten())

# Fully-connected layer
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam') # loss - mean squared error - mean absolute error

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')
