# Import required libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,AveragePooling2D,BatchNormalization
from data_classification import *

# Define input size and number of classes
input_size = (227, 227)
num_classes = 2


# Define the model
# Define the model
model = Sequential([
    Input(shape=input_size+(3,)),
    Conv2D(32, 3, activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2),strides=2),
    Conv2D(64, 3, activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2),strides=2),
    Conv2D(256, 3, activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2),strides=2),
    Conv2D(512, 3, activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2),strides=2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])



if __name__ == '__main__':
    print(model.summary())
    print(input_size+(3,))
    


