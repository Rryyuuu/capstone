from PIL import Image
from data_classification import *
import os
from PIL import Image


# Save the reassembled image

if __name__=='__main__':

    image_paths = drone_heatmap_dir
    output_path ="C:/Users/ryu/Desktop/main_data/gis/heatmap/combine_heatmap_2.jpg"
    original_path = "C:/Users/ryu/Desktop/main_data/gis/Drone_image/drone_image_5.tif"

    original_image = Image.open(original_path)
    filepath_list = [os.path.join(image_paths,filename) for filename in os.listdir(image_paths)]
    patches = [Image.open(path) for path in filepath_list]
    patch_width , patch_height =  patches[0].size
    width ,height = original_image.size 
    # Create a new blank image to reassemble the patches
    reassembled_image = Image.new('RGB', (width, height))
    
    # Assemble the patches back into the new image
    x_offset = 0
    y_offset = 0
    for patch in patches:
        reassembled_image.paste(patch, (x_offset, y_offset))
        y_offset += patch_height
        if y_offset >= height:
            y_offset = 0
            x_offset += patch_width

    reassembled_image.save(output_path)  # Replace 'reassembled_image.jpg' with your desired output file name
