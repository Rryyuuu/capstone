from PIL import Image
import numpy as np
from data_classification import drone_image_dir, drone_image_small_dir
import os

if __name__ == '__main__':

    drone_image_path = os.path.join(drone_image_dir,'drone_image_6.tif')

    # 이미지 파일 열기
    img = Image.open(drone_image_path)

    # 이미지 데이터 numpy 배열로 변환
    img_data = np.array(img)

    # 이미지 크기 출력
    print('Image size:', img_data.shape)





    # 이미지 크기 가져오기
    width, height = img.size

    # 각각의 타일 크기 계산
    tile_width = 227
    tile_height = 227

    # 타일의 개수 계산
    num_tiles_width = int(width / tile_width)
    num_tiles_height = int(height / tile_height)
    
    os.mkdir(drone_image_small_dir)
    # 타일 분할 시작
    for i in range(num_tiles_width):
        for j in range(num_tiles_height):
            # 타일의 위치 계산
            x = i * tile_width
            y = j * tile_height
            # 타일 분할
            tile = img.crop((x, y, x+tile_width, y+tile_height))
            # 타일 저장
            drone_image_small_path = os.path.join(drone_image_small_dir,f'tile_{i}_{j}.tif')
            
            tile.save(drone_image_small_path)
