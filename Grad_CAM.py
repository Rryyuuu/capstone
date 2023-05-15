import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from keras.models import load_model, Model
from keras.utils import load_img, img_to_array,array_to_img
from keras.applications.vgg16 import preprocess_input
from tensorflow import GradientTape, function,argmax,reduce_mean,reduce_sum,multiply,maximum,reduce_max
from data_classification import model_save_dir, drone_image_small_dir, validation_positive_dir,drone_image_test_positive_dir
from PIL import Image
from CNN_layers import input_size
import keras
from keras.preprocessing.image import ImageDataGenerator

def get_img_array(img_path, size):
    # Load image and convert to numpy array
    img = load_img(img_path, target_size=size)
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor/=255.

    return img_tensor

if __name__=='__main__':
    # Load the trained model
    model = load_model(model_save_dir)

    # Specify the layer name to use for Grad-CAM
    layer_name = 'conv2d_11'
    # img_name = 'tile_15_11.tif'
    for filename in os.listdir(drone_image_small_dir):
    # Load the test image
        img_path = os.path.join(drone_image_small_dir,filename)
        
        img_array = get_img_array(img_path, input_size)
        
        pred = model.predict(img_array)
        
        try:
            if pred[0][0]<pred[0][1]:
                class_idx = np.argmax(pred[0])
                class_output = model.output[:, class_idx]
                last_conv_layer = model.get_layer(layer_name)
                grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
                
                with GradientTape() as tape:
                    conv_output, preds = grad_model(img_array)
                    loss = preds[:, class_idx]
                    grads = tape.gradient(loss, conv_output)[0]

                # Compute the channel-wise mean of the gradients and the feature maps
                weights = reduce_mean(grads, axis=(0, 1))
                cam = np.dot(conv_output[0], weights.numpy())

                cam = cv2.resize(cam, (img_array.shape[2], img_array.shape[1]))
                cam = np.maximum(cam, 0)
                cam = cam / cam.max()
                
                plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
                plt.imshow(cam, cmap='jet', alpha=0.5)
                plt.savefig('C:/Users/ryu/Desktop/main_data/gis/drone_heatmap/{}.png'.format(filename))
                # plt.show()
            else:
                plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
                plt.savefig('C:/Users/ryu/Desktop/main_data/gis/drone_heatmap/{}.png'.format(filename))
        except:
            plt.imshow(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB))
            plt.savefig('C:/Users/ryu/Desktop/main_data/gis/drone_heatmap/{}.png'.format(filename))
            
                