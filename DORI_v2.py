from FastSAM.fastsam import *
import cv2
import os
import shutil
from utils import compute_blob_mean_and_covariance
import numpy as np
import matplotlib.pyplot as plt
from utils import plotErrorEllipse
import skimage
from ultralytics import SAM
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel

import warnings
warnings.filterwarnings('ignore')

# Specify path to checkpoint. (If checkpoint does not exist, the implementation in FastSAM repo downloads it.)
#fastSamModel = FastSAM('./FastSAM/Models/FastSAM-x.pt')
fastSamModel = SAM('sam_b.pt')
DEVICE = 'cuda'





# Define the file location !!! Might have to change this
file_location = 'fish_img'

# Get all image file names in the directory
image_files = [f for f in os.listdir(file_location) if f.endswith('.jpg') or f.endswith('.png')]

# Setup output folder
outputFolder = 'potential_fish_v2/'

# Check if output folder exists and create it if necessary
if (not os.path.isdir(outputFolder)):
    os.makedirs(outputFolder)

# Sort the image files by their filenames such that frame0 then frame1 then frame2 ... then frame 10 then 11 then 12, etc
image_files.sort(key=lambda x: int(x.split('.')[0].split('e')[-1]))

image_files = image_files[200:2000]

# Initialize an empty list to store the average luminance of each superpixel
average_luminance = []

# initialize count
count = 0

potential_fish_frames = []

# For FastSam:
# Specify confidence and IoU parameters (see FastSAM paper or rather YOLO v8 documentation)
conf = 0.5
iou = 0.9

# Loop through each image file
for image_file in image_files:
    # Read the image
    image = cv2.imread(os.path.join(file_location, image_file))

    luminance = np.mean(image)
    count += 1

    if count < 50:
        average_luminance.append(luminance)
        continue
    
    # Append the average luminance to the list
    average_luminance.append(luminance)

    luminance_diff = abs(luminance - abs(luminance - np.mean(average_luminance[count-50:count])))

    if luminance_diff > 1:
        potential_fish_frames.append(count)

        # Run FastSAM
        everything_results = fastSamModel(image, device=DEVICE, retina_masks=True, imgsz=1024, conf=conf, iou=iou,)
        prompt_process = FastSAMPrompt(image, everything_results, device=DEVICE)
        segmask = prompt_process.everything_prompt()
        # print(len(segmask))

        # Segment the images for better classification
        image_array = []
        for mask_index in range(len(segmask)): # Append all segments for classifications
            # Masked out the segmentation
            potential_fish_img = image.copy()
            potential_fish_img[~segmask[mask_index,:,:],:] = 0

            # Convert to PIL Image
            image_pil = Image.fromarray(potential_fish_img)
            image_array.append(image_pil)

        # Initialize the CLIP model and processor
        model = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        # Classification labels
        classes = ['fish', 'aquatic plants', 'trash', 'water', 'land', 'ocean', 'submarine']
        
        # Process the image for CLIP (resize, convert to tensor, normalize)
        inputs = processor(text=classes, images=image_array, return_tensors="pt", padding=True)

        # Forward pass: get the image features from CLIP
        outputs = model(**inputs)

        # Classification:
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1) 
        classification_outputs = torch.argmax(probs, dim=1)
        # print(torch.argmax(probs, dim=1))

        # If is classified as 'fish'
        if torch.any(classification_outputs==0):
            # Save the image that is detected and classified with fish in the output Folder
            cv2.imwrite(outputFolder+image_file, image) 
            # print('fish')
            # print(image_file)
            # plt.imshow(image)



# Process the image for CLIP (resize, convert to tensor, normalize)
# inputs = processor(text=classes, images=[image_pil], return_tensors="pt", padding=True)

# # Print the average luminance of each superpixel
# for i, luminance in enumerate(average_luminance):
#     print(f"Superpixel {i+1}: Average Luminance = {luminance}")
        
# print(potential_fish_frames)
