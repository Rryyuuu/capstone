import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from keras.models import load_model, Model
from keras.utils import load_img, img_to_array,array_to_img
from keras.applications.vgg16 import preprocess_input
from tensorflow import GradientTape, function,argmax,reduce_mean,reduce_sum,multiply,maximum,reduce_max
from data_classification import model_save_dir, drone_image_small_dir, validation_positive_dir
from PIL import Image
from CNN_layers import input_size
import keras


def get_img_array(img_path, size):
    # Load image and convert to numpy array
    img = load_img(img_path, target_size=size)
    img_array = img_to_array(img)
    # Expand dimensions to match input shape of model
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

if __name__=='__main__':
    # Load the trained model
    model = load_model(model_save_dir)

    # Specify the layer name to use for Grad-CAM
    layer_name = 'conv2d_11'

    # Load the test image
    img_path = os.path.join(validation_positive_dir,'19000.jpg')
    
    img_array = get_img_array(img_path, input_size)
    
    preds = model.predict(img_array)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer(layer_name)
    grad_model = Model([model.inputs], [last_conv_layer.output, model.output])
    
    with GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        grads = tape.gradient(preds, conv_output)[0]

    # Compute the channel-wise mean of the gradients and the feature maps
    weights = reduce_mean(grads, axis=(0, 1))
    cam = reduce_sum(multiply(weights, conv_output), axis=-1)

    # Resize the heatmap to the original image size and normalize it
    heatmap = cv2.resize(cam.numpy(), (input_size[1], input_size[0]))
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / np.max(heatmap)
    heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2GRAY)
    
    '''
    # Apply the heatmap to the original image
    img = cv2.imread(img_path)
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Plot the results
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

    plt.imshow(superimposed_img)
    plt.title('Grad-CAM')
    plt.axis('off')
    plt.show()
    '''
    # Plot the heatmap on top of the original image
    img = plt.imread(img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.show()
