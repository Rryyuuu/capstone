from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_classification import test_dir,model_save_dir,drone_image_test_dir,validation_dir
from CNN_layers import input_size
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from sklearn.metrics import confusion_matrix
import os

if __name__ == '__main__':
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(validation_dir, target_size=input_size, batch_size=1,class_mode='categorical')

    y_true=[]

    for i in range(len(test_generator.filenames)):
        # Get file path and directory name
        file_path = test_generator.filepaths[i]
        dir_name = os.path.dirname(file_path)
        # Get file name and new file path
        file_layer = os.path.basename(dir_name)
        if file_layer == 'Positive':
            y_true.append(1)
        else:
            y_true.append(0)



    # Assuming you have already trained and saved your CNN model
    model = load_model(model_save_dir)
    y_evaluate = model.evaluate(test_generator)
    print(y_evaluate)
    y_pred = model.predict(test_generator)
    y_pred = np.argmax(y_pred,axis=1)
    Binary_Confusion_Matrix= confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(Binary_Confusion_Matrix)
    print('True positives: ', tp)
    print('True negatives: ', tn)
    print('False positives: ', fp)
    print('False negatives: ', fn)



