import os, shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#main dir
original_dataset_positive_dir = 'C:/Users/ryu/Desktop/main_data/gis/Concrete_Crack_Images/Positive'
original_dataset_negative_dir = 'C:/Users/ryu/Desktop/main_data/gis/Concrete_Crack_Images/Negative'

#sub dir
base_dir = 'C:/Users/ryu/Desktop/main_data/gis/Concrete_Crack_Images_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')


train_positive_dir = os.path.join(train_dir, 'Positive')
train_negative_dir = os.path.join(train_dir,'Negative')

validation_positive_dir = os.path.join(validation_dir, 'Positive')
validation_negative_dir = os.path.join(validation_dir, 'Negative')

test_positive_dir = os.path.join(test_dir, 'Positive')
test_negative_dir = os.path.join(test_dir, 'Negative')


model_save_dir ='C:/Users/ryu/Desktop/main_data/gis/model_save/crack_detection.h5'

drone_image_dir = 'C:/Users/ryu/Desktop/main_data/gis/Drone_image'
drone_image_small_dir = 'C:/Users/ryu/Desktop/main_data/gis/Drone_image_small'
drone_image_test_dir = 'C:/Users/ryu/Desktop/main_data/gis/Drone_image_test'


if __name__ == '__main__':
  os.mkdir(base_dir)
  os.mkdir(train_dir)
  os.mkdir(validation_dir)
  os.mkdir(test_dir)
  os.mkdir(train_positive_dir)
  os.mkdir(train_negative_dir)
  os.mkdir(validation_positive_dir)
  os.mkdir(validation_negative_dir)
  os.mkdir(test_positive_dir)
  os.mkdir(test_negative_dir)
  
  #rename the file
  '''
  fnames = ['{0:05d}_1.jpg'.format(i) for i in range(10000,19379)]
  i=10000
  for fname in fnames:
    src = os.path.join(original_dataset_positive_dir, fname)
    dst = str(i)+'.jpg'
    dst = os.path.join(original_dataset_positive_dir,dst)
    os.rename(src,dst)
    i+=1
  '''


  fnames = ['{0:05d}.jpg'.format(i) for i in range(1,12001)]
  for fname in fnames:
    src = os.path.join(original_dataset_positive_dir, fname)
    dst = os.path.join(train_positive_dir, fname)
    shutil.copyfile(src, dst)


  fnames = ['{0:05d}.jpg'.format(i) for i in range(12001,16001)]
  for fname in fnames:
    src = os.path.join(original_dataset_positive_dir, fname)
    dst = os.path.join(validation_positive_dir, fname)
    shutil.copyfile(src, dst)

  fnames = ['{0:05d}.jpg'.format(i) for i in range(16001,20001)]
  for fname in fnames:
    src = os.path.join(original_dataset_positive_dir, fname)
    dst = os.path.join(test_positive_dir, fname)
    shutil.copyfile(src, dst)
    
  fnames = ['{0:05d}.jpg'.format(i) for i in range(1,12001)]
  for fname in fnames:
    src = os.path.join(original_dataset_negative_dir, fname)
    dst = os.path.join(train_negative_dir, fname)
    shutil.copyfile(src, dst)
    
    
  fnames = ['{0:05d}.jpg'.format(i) for i in range(12001,16001)]
  for fname in fnames:
    src = os.path.join(original_dataset_negative_dir, fname)
    dst = os.path.join(validation_negative_dir, fname)
    shutil.copyfile(src, dst)
    
  fnames = ['{0:05d}.jpg'.format(i) for i in range(16001,20001)]
  for fname in fnames:
    src = os.path.join(original_dataset_negative_dir, fname)
    dst = os.path.join(test_negative_dir, fname)
    shutil.copyfile(src, dst)
    
    
    
    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  




