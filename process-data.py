import os
import time

import pandas as pd
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage import binary_opening, generate_binary_structure, binary_fill_holes
from segment_anything import sam_model_registry, SamPredictor
import json

data = pd.DataFrame()
data["imageid"] = ""
data["district"] = ""
data["name"] = ""
data["latitude"] = ""
data["longitude"] = ""
data["height"] = ""
data["recordedat"] = ""
data["svf"] = ""


def show_mask(mask, root, file, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

    mask_8bit = (mask * 255).astype(np.uint8)
    binary_mask = np.where(mask_8bit > 0.5 * 255, 255, 0).astype(np.uint8)
    filled_image = binary_fill_holes(binary_mask)
    structuring_element = generate_binary_structure(2, 1)
    cleaned_image = binary_opening(filled_image, structure=structuring_element)
    cleaned_image_converted = Image.fromarray(np.uint8(cleaned_image * 255))

    mask_img = Image.open("mask.png")
    binary_img = cleaned_image_converted
    binary_img = binary_img.convert("RGBA")
    alpha_channel = mask_img.getchannel('A')
    output_img = Image.composite(binary_img, mask_img, alpha_channel)
    output_img.save(root + "/" + file.strip(".png") + "_mask.png")

    # Convert image to grayscale and then to binary (white pixels as sky, black as non-sky)
    gray_image = output_img.convert('L')
    binary_image = np.array(gray_image) > 128  # assuming white (sky) is >128 in grayscale

    # Get image dimensions and center
    h, w = binary_image.shape
    center_x, center_y = w // 2, h // 2
    radius = min(center_x, center_y)

    # Create a grid of distances from the center
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    # Number of rings and initialize SVF calculation
    n = 36
    rings = np.linspace(0, radius, n + 1)
    svf = 0

    # Calculate proportions and SVF<
    for i in range(1, n + 1):
        mask = (dist_from_center >= rings[i - 1]) & (dist_from_center < rings[i])
        if np.any(mask):
            sky_pixels = np.sum(binary_image[mask])
            total_pixels = np.sum(mask)
            p_i = sky_pixels / total_pixels
            svf += np.sin(np.pi * (2 * i - 1) / (2 * n)) * p_i

    # Scaling factor for SVF
    svf *= (np.pi / (2 * n))
    data.loc[index, 'svf'] = svf
    # Output the computed SVF
    print(f"Calculated SVF: {svf}")


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# directory = r"C:\Users\sukruburak.cetin\Desktop\panorama-tiles-to-projections\data_fatih"
directory = r"C:\Users\sukruburak.cetin\Desktop\panorama-tiles-to-projections\data_done\data_all"

# Count total files
total_files = sum([len([f for f in files if f.endswith("_fisheye.png")]) for r, d, files in os.walk(directory)])
print("total_file_count: ", total_files)

# Create a progress bar
with tqdm(total=total_files) as pbar:
    for root, dirs, files in os.walk(directory):
        for index, file in enumerate(files):
            if file.endswith("_properties.json"):
                with open(root + "\\" + file) as json_file:
                    json_data = json.load(json_file)
                    data.loc[index, 'imageid'] = json_data["imageid"]
                    # data.loc[index, 'district'] = json_data["ILCE_ADI_resolved"]
                    data.loc[index, 'name'] = json_data["AD"]
                    data.loc[index, 'latitude'] = json_data["latitude"]
                    data.loc[index, 'longitude'] = json_data["longitude"]
                    data.loc[index, 'height'] = json_data["height"]
                    data.loc[index, 'recordedat'] = json_data["recordedat"]

            if file.endswith("_fisheye.png"):
                start_time = time.time()

                img = Image.open(root + "/" + file)
                width, height = img.size
                new_width = height
                left = (width - new_width) / 2
                right = width - left
                img_cropped = img.crop((left, 0, right, height))
                cropped_image_path = root + "/" + file.strip(".png") + "_cropped.png"
                img_cropped.save(cropped_image_path)

                img_cropped_final = cv2.imread(cropped_image_path)
                image = cv2.cvtColor(img_cropped_final, cv2.COLOR_BGR2RGB)

                mask_predictor = SamPredictor(sam)
                mask_predictor.set_image(image)
                input_point = np.array([[512, 512]])
                input_label = np.array([1])

                masks, scores, logits = mask_predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )

                fig, ax = plt.subplots()
                show_mask(masks[0], root, file, ax)
                plt.close(fig)

                end_time = time.time()
                duration = end_time - start_time
                print(f"\nProcessed {file} in {duration:.2f} seconds")
                data.to_csv(r"kadikoy_results.csv", mode='a', index=False, header=None)
                pbar.update(1)

print("end")
