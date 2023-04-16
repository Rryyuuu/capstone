# Import required libraries
from keras.preprocessing.image import ImageDataGenerator
from data_classification import *
from CNN_layers import *



# Define data generators for training and validation data
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=input_size, batch_size=50,class_mode='categorical')
val_generator = val_datagen.flow_from_directory(validation_dir, target_size=input_size, batch_size=50,class_mode='categorical')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, steps_per_epoch=40 ,epochs=20, validation_data=val_generator, validation_steps=40)

model.save(model_save_dir)