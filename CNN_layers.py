# Import required libraries
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense,BatchNormalization,Dropout
from data_classification import *

# Define input size and number of classes
input_size = (227, 227)
num_classes = 2


# Define the model

model = Sequential([
    Input(shape=input_size+(3,)),
    Conv2D(32, 3, activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2),strides=2),
    Conv2D(64, 3, activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2),strides=2),
    Conv2D(128, 3, activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2),strides=2),
    Conv2D(256, 3, activation='relu',padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2),strides=2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])



if __name__ == '__main__':
    print(model.summary())
    print(input_size+(3,))
    


