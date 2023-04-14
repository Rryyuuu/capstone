from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_classification import test_dir,test_positive_dir,test_negative_dir,model_save_dir
from CNN_layers import input_size
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array

if __name__ == '__main__':
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=input_size, batch_size=4,class_mode='categorical')

    # Assuming you have already trained and saved your CNN model
    model = load_model(model_save_dir)
    y_pred = model.evaluate(test_generator)
    print(y_pred)




