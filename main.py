import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import slic
from skimage.color import label2rgb

# Define the file location
file_location = '/home/annika/data/fish_data/fish_img'

# Get all image file names in the directory
image_files = [f for f in os.listdir(file_location) if f.endswith('.jpg') or f.endswith('.png')]

# Sort the image files by their filenames such that frame0 then frame1 then frame2 ... then frame 10 then 11 then 12, etc
image_files.sort(key=lambda x: int(x.split('.')[0].split('e')[-1]))

image_files = image_files[200:1000]

# Initialize an empty list to store the average luminance of each superpixel
average_luminance = []

# initialize count
count = 0

potential_fish_frames = []

# Loop through each image file
for image_file in image_files:
    # Read the image
    image = cv2.imread(os.path.join(file_location, image_file))
    
    # # num_segments is the desired number of superpixels
    # segments = slic(image, n_segments=100, compactness=10, sigma=1)

    # # Create an image showing the superpixel boundaries
    # segmented_image = label2rgb(segments, image, kind='avg')

    # # average luminance of each superpixel
    # luminance = np.mean(segmented_image)

    luminance = np.mean(image)
    count += 1

    if count < 50:
        average_luminance.append(luminance)
        continue
    
    # Append the average luminance to the list
    average_luminance.append(luminance)

    luminance_diff = abs(luminance - average_luminance[count-50])

    if luminance_diff > 1:
        potential_fish_frames.append(count)

# # Print the average luminance of each superpixel
# for i, luminance in enumerate(average_luminance):
#     print(f"Superpixel {i+1}: Average Luminance = {luminance}")
        
print(potential_fish_frames)
