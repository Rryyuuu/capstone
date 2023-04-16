import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model,Model
from data_classification import model_save_dir,drone_image_small_dir,test_positive_dir
from keras.utils import load_img, img_to_array,array_to_img
from keras.backend import function
import os
from CNN_layers import input_size
from keras.applications.vgg16 import preprocess_input
from PIL.Image import fromarray,LANCZOS
from tensorflow import GradientTape,argmax,reduce_mean,Variable,reduce_max,multiply,convert_to_tensor,identity,cast,reduce_sum,image,maximum
from keras import backend as K




# Load the trained model
model = load_model(model_save_dir)


# Specify the layer name to use for Grad-CAM
layer_name = 'conv2d_2'

img_path = os.path.join(test_positive_dir,'18000.jpg')
img = load_img(img_path, target_size=input_size)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
class_idx = np.argmax(preds[0])

last_conv_layer = model.get_layer(layer_name)

grads = K.gradients(model.output[:, class_idx], last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))

heatmap_fn = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])


# Plot the original image and the CAM heatmap
plt.imshow(img)
plt.imshow(cam, cmap='jet', alpha=0.5)
plt.show()








'''
# Specify the layer name to use for Grad-CAM
layer_name = 'conv2d_2'

img = load_img(img_path,target_size=input_size)

x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)

crack_index = np.argmax(preds[0])
print(crack_index)
crack_output = model.output[:,crack_index]
print(crack_output)
last_conv_layer = model.get_layer('conv2d_2')
print(last_conv_layer)
grads = K.gradients(crack_output,last_conv_layer.output)
print(grads)

pooled_grads = K.function(grads, axis=(0,1,2))


iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([0])

for i in range(128):
    conv_layer_output_value[:, :, i]*= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis =-1) 


heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)   

'''
