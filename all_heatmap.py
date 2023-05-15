from PIL import Image
from drone_preprocessing import num_tiles_width, num_tiles_height
from data_classification import *
import os

if __name__=='__main__':
    # Define the number of rows and columns
    rows = num_tiles_height
    columns = num_tiles_width

    image_path = drone_image_dir
    image_name = os.path.join(image_path, 'drone_image_5')
    # Open and get the size of the first sub-image
    sub_image = Image.open("sub_image_0.jpg")
    sub_width, sub_height = sub_image.size

    # Calculate the total width and height of the combined image
    total_width = sub_width * columns
    total_height = sub_height * rows

    # Create a new blank image with the size of the combined image
    combined_image = Image.new("RGB", (total_width, total_height))

    # Iterate through the sub-images and paste them onto the combined image
    for i in range(rows):
        for j in range(columns):
            index = i * columns + j
            sub_image_path = f"sub_image_{index}.jpg"
            sub_image = Image.open(sub_image_path)
            combined_image.paste(sub_image, (j * sub_width, i * sub_height))

    # Save the merged image
    combined_image.save("merged_image.jpg")
