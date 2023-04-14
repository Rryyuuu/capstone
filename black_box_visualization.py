from keras.models import load_model
from keras import models
from data_classification import *
from CNN_layers import input_size
from keras.utils import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os





    
if __name__ == '__main__':
    model = load_model(model_save_dir)
    model.summary()

    img_path =os.path.join(train_positive_dir,'05000.jpg')
    img = load_img(img_path,target_size = input_size)
    img_tensor = img_to_array(img)
    print(img_tensor.shape)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor/=255.

    layer_outputs = [layer.output for layer in model.layers[:9]]
    activation_model = models.Model(inputs = model.input, outputs = layer_outputs)

    activations = activation_model.predict(img_tensor)
    # first_layer_activation = activations[0]
    # plt.matshow(first_layer_activation[0, :, :, 19], cmap='viridis')
  
    
    layer_names = []
    for layer in model.layers[:9]:
        layer_names.append(layer.name)

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        n_featurs = layer_activation.shape[-1]    
        
        size = layer_activation.shape[1]
        
        n_cols = n_featurs // images_per_row
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :,col*images_per_row]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 227).astype('uint8')
                display_grid[col * size : (col + 1) * size,
                            row * size : (row + 1) * size] = channel_image

        scale = 1. / size
        plt.figure(figsize= (scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap = 'viridis')
        plt.savefig('C:/Users/ryu/Desktop/main_data/gis/activation_layers/{0}.png'.format(layer_name))
            
    plt.show()



