from PIL import Image
import numpy as np
from data_classification import drone_image_path

# 이미지 파일 열기
img = Image.open(drone_image_path)

# 이미지 데이터 numpy 배열로 변환
img_data = np.array(img)

# 이미지 크기 출력
print('Image size:', img_data.shape)
