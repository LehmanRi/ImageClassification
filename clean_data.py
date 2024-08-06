# import os
# import cv2
# import numpy as np
#
# # Path to the directory containing training images
# train_dir = 'dataset/train'
#
# # Create a directory to save the cleaned images
# cleaned_dir = 'dataset/cleaned'
# if not os.path.exists(cleaned_dir):
#     os.makedirs(cleaned_dir)
#
#
# # Function to preprocess images
# def preprocess_image(image_path, mean_pixel):
#     # Load image
#     img = cv2.imread(image_path)
#     if img is None:
#         return None
#
#     # Resize image to 256x256
#     img = cv2.resize(img, (256, 256))
#
#     # Center-crop to 224x224
#     h, w, _ = img.shape
#     startx = w // 2 - (224 // 2)
#     starty = h // 2 - (224 // 2)
#     img = img[starty:starty + 224, startx:startx + 224]
#
#     # Normalize the image by subtracting the mean pixel value
#     img = img.astype(np.float32)
#     img -= mean_pixel
#
#     return img
#
#
# # Calculate mean pixel value from the dataset
# def calculate_mean_pixel(image_dir):
#     mean_pixel = np.zeros(3)
#     count = 0
#     for subdir in os.listdir(image_dir):
#         subdir_path = os.path.join(image_dir, subdir)
#         if os.path.isdir(subdir_path):
#             for filename in os.listdir(subdir_path):
#                 image_path = os.path.join(subdir_path, filename)
#                 img = cv2.imread(image_path)
#                 if img is not None:
#                     mean_pixel += np.mean(img, axis=(0, 1))
#                     count += 1
#     mean_pixel /= count
#     return mean_pixel
#
#
# # Calculate the mean pixel value of the training set
# mean_pixel = calculate_mean_pixel(train_dir)
#
# # Preprocess and save each image
# for subdir in os.listdir(train_dir):
#     subdir_path = os.path.join(train_dir, subdir)
#     if os.path.isdir(subdir_path):
#         cleaned_subdir_path = os.path.join(cleaned_dir, subdir)
#         if not os.path.exists(cleaned_subdir_path):
#             os.makedirs(cleaned_subdir_path)
#         for filename in os.listdir(subdir_path):
#             image_path = os.path.join(subdir_path, filename)
#             cleaned_image = preprocess_image(image_path, mean_pixel)
#             if cleaned_image is not None:
#                 cleaned_image_path = os.path.join(cleaned_subdir_path, filename)
#                 cv2.imwrite(cleaned_image_path, cleaned_image)
#
# print("Data cleaning and preprocessing completed.")


import os
import cv2
import numpy as np

# Path to the directory containing training images
train_dir = 'dataset/train'

# Create a directory to save the cleaned images
cleaned_dir = 'dataset/cleaned'
if not os.path.exists(cleaned_dir):
    os.makedirs(cleaned_dir)

# Function to preprocess images according to the article's requirements
def preprocess_image(image_path, mean_pixel):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Get original dimensions
    h, w, _ = img.shape

    # Determine new dimensions while maintaining aspect ratio
    if h < w:
        new_h = 256
        new_w = int(w * (256 / h))
    else:
        new_w = 256
        new_h = int(h * (256 / w))

    # Resize the image such that the shorter side is 256 pixels
    img = cv2.resize(img, (new_w, new_h))

    # Center-crop to 256x256
    h, w, _ = img.shape
    startx = w // 2 - (256 // 2)
    starty = h // 2 - (256 // 2)
    img = img[starty:starty + 256, startx:startx + 256]

    # Subtract the mean pixel value
    img = img.astype(np.float32)
    img -= mean_pixel

    return img

# Calculate mean pixel value from the dataset
def calculate_mean_pixel(image_dir):
    mean_pixel = np.zeros(3)
    count = 0
    for subdir in os.listdir(image_dir):
        subdir_path = os.path.join(image_dir, subdir)
        if os.path.isdir(subdir_path):
            for filename in os.listdir(subdir_path):
                image_path = os.path.join(subdir_path, filename)
                img = cv2.imread(image_path)
                if img is not None:
                    mean_pixel += np.mean(img, axis=(0, 1))
                    count += 1
    mean_pixel /= count
    return mean_pixel

# Calculate the mean pixel value of the training set
mean_pixel = calculate_mean_pixel(train_dir)

# Preprocess and save each image
for subdir in os.listdir(train_dir):
    subdir_path = os.path.join(train_dir, subdir)
    if os.path.isdir(subdir_path):
        cleaned_subdir_path = os.path.join(cleaned_dir, subdir)
        if not os.path.exists(cleaned_subdir_path):
            os.makedirs(cleaned_subdir_path)
        for filename in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, filename)
            cleaned_image = preprocess_image(image_path, mean_pixel)
            if cleaned_image is not None:
                cleaned_image_path = os.path.join(cleaned_subdir_path, filename)
                cv2.imwrite(cleaned_image_path, cleaned_image)

print("Data cleaning and preprocessing completed.")
