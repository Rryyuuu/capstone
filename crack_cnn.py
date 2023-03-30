# Import required libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from data_classification import *
from CNN_layers import *



# Define data generators for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=input_size, batch_size=50, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(validation_dir, target_size=input_size, batch_size=50, class_mode='categorical')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=30, validation_data=val_generator)

model.save('C:/Users/ryu/Desktop/main_data/gis/model_save/crack_detection.h5')