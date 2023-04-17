import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from keras.models import load_model, Model
from keras.utils import load_img, img_to_array,array_to_img
from keras.applications.vgg16 import preprocess_input
from tensorflow import GradientTape, function
from data_classification import model_save_dir, drone_image_small_dir, test_positive_dir
from PIL import Image
from CNN_layers import input_size
import keras
# Load the trained model
model = load_model(model_save_dir)


# Specify the layer name to use for Grad-CAM
layer_name = 'conv2d_5'

# Load the test image
img_path = os.path.join(test_positive_dir,'18000.jpg')
img = load_img(img_path, target_size=input_size)  # assuming input_size=(224, 224)

# Convert the image to a numpy array and preprocess it
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make a prediction on the image
preds = model.predict(x)

# Get the index of the predicted class
crack_index = np.argmax(preds[0])
print(crack_index)
# Get the output of the predicted class
crack_output = model.output[:, crack_index]
print('this is output',crack_output)
# Get the last convolutional layer of the model
last_conv_layer = model.get_layer(layer_name)
print(last_conv_layer)


'''
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam()

with GradientTape() as tape:
    # Make a forward pass
    y_pred = model(x)
    # Compute loss
    loss = loss_fn(1, y_pred)
    print(loss)
# Compute gradients
grads = tape.gradient(loss, model.trainable_variables)

# Print gradients
for var, grad in zip(model.trainable_variables, grads):
    print(f"Variable name: {var.name}")
    print(f"Gradient values: {grad}\n")
'''
# Compute the gradients of the predicted class with respect to the last conv layer
with GradientTape() as tape:
    grads = tape.gradient(last_conv_layer.output,crack_output)[0]
print(grads)
# Compute the mean of the gradients over each feature map
pooled_grads = function([model.input], [grads, last_conv_layer.output])[0]

# Multiply each feature map by its corresponding gradient
iterate = function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(conv_layer_output_value.shape[-1]):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

# Create the heatmap by taking the mean of the resulting feature maps
heatmap = np.mean(conv_layer_output_value, axis=-1)

# Normalize the heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# Plot the heatmap
plt.matshow(heatmap)
