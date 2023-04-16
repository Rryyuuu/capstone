import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
from data_classification import model_save_dir,drone_image_small_dir,test_positive_dir
from keras.utils import load_img, img_to_array
from keras.backend import function
import os
from CNN_layers import input_size

# 이미지 경로 설정
img_path = os.path.join(test_positive_dir,'18000.jpg')
print(img_path)
# 이미지 불러오기
img = load_img(img_path, target_size=input_size)
print(img)
# 이미지를 배열로 변환
x = img_to_array(img)

# 이미지 전처리
x = np.expand_dims(x, axis=0)
x/=255.


#  모델 불러오기
model = load_model(model_save_dir)

# 마지막 convolutional layer 가져오기
last_conv_layer = model.get_layer('conv2d_2')

# 모델의 출력값 가져오기
preds = model.predict(x)
print(preds)
# 클래스 인덱스 가져오기
class_idx = np.argmax(preds[0])
print(class_idx)
# 클래스별 가중치 가져오기
class_output = model.output[:, class_idx]

# 마지막 convolutional layer의 출력값과 클래스별 가중치를 곱하기
grads = function([model.input], [last_conv_layer.output, class_output])([x])
conv_output, class_weights = grads[0], grads[1]
cam = np.zeros((conv_output.shape[1], conv_output.shape[2]), dtype=np.float32)

# 가중치와 convolutional layer 출력값을 곱하여 heatmap 생성
for i, w in enumerate(class_weights[0]):
    cam += w * conv_output[0, :, :, i]

# heatmap을 0~1 범위로 정규화
cam /= np.max(cam)

# heatmap 리사이즈
cam = cv2.resize(cam, input_size)

# heatmap을 원본 이미지에 적용
heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

# heatmap과 원본 이미지 결합
superimposed_img = cv2.addWeighted(cv2.cvtColor(np.uint8(x[0]), cv2.COLOR_RGB2BGR), 0.6, heatmap, 0.4, 0)

# 결과 이미지 출력
plt.imshow(superimposed_img)
plt.show()

