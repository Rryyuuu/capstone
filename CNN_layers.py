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
    Conv2D(16, 5, activation='relu'),
    Conv2D(16, 5, activation='relu'),
    Conv2D(16, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling2D((5, 5),strides=1),
    Conv2D(32, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling2D((5, 5),strides=2),
    Conv2D(32, 5, activation='relu'),
    Conv2D(32, 5, activation='relu'),
    Conv2D(32, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling2D((5, 5),strides=1),
    Conv2D(64, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling2D((5, 5),strides=2),
    Conv2D(64, 5, activation='relu'),
    Conv2D(64, 5, activation='relu'),
    Conv2D(64, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling2D((5, 5),strides=1),
    Conv2D(128, 5, activation='relu'),
    BatchNormalization(),
    MaxPooling2D((5, 5),strides=2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')
])



if __name__ == '__main__':
    print(model.summary())
    print(input_size+(3,))
    


