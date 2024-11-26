import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the data augmentation parameters
datagen = ImageDataGenerator(
    horizontal_flip=True,  # Horizontal reflection
    fill_mode='nearest'  # Filling strategy for shifted patches
)


def resize_image(img, target_size=(256, 256)):
    """Resizes image to target size."""
    return img.resize(target_size, Image.Resampling.LANCZOS)

def extract_random_patches(img, target_size=(224, 224), num_patches=5):
    """Extracts multiple random patches from a resized image, handling both grayscale and RGB images."""
    if len(img.shape) == 2:  # Check if the image is grayscale
        img = np.stack((img,) * 3, axis=-1)  # Convert grayscale to RGB by duplicating the channels

    patches = []
    h, w, _ = img.shape
    for _ in range(num_patches):
        top = np.random.randint(0, h - target_size[0] + 1)
        left = np.random.randint(0, w - target_size[1] + 1)
        patch = img[top:top + target_size[0], left:left + target_size[1], :]
        patches.append(patch)
    return patches

def augment_patches(patches):
    """Applies horizontal flip to each patch and returns original and flipped patches."""
    augmented_images = []
    for patch in patches:
        original = patch
        flipped = np.fliplr(patch)  # Horizontal flip
        augmented_images.append(original)
        augmented_images.append(flipped)
    return augmented_images

def process_dataset(dataset_dir, output_dir):
    """Processes all images in the dataset, resizes them, extracts patches, and applies augmentation."""
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if os.path.isdir(class_path):
            output_class_dir = os.path.join(output_dir, class_dir)
            os.makedirs(output_class_dir, exist_ok=True)

            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path)
                resized_img = resize_image(img)

                # Convert the PIL image to a numpy array
                resized_img = np.array(resized_img)
#
                # Extract random patches
                patches = extract_random_patches(resized_img, num_patches=5)

                # Augment patches by flipping horizontally
                augmented_images = augment_patches(patches)

                # Save augmented images to output directory
                for i, aug_img in enumerate(augmented_images):
                    aug_img_pil = Image.fromarray(aug_img.astype('uint8'))

                    # Check if the image is in RGBA mode and convert it to RGB before saving
                    if aug_img_pil.mode == 'RGBA':
                        aug_img_pil = aug_img_pil.convert('RGB')

                    aug_img_path = os.path.join(output_class_dir, f'{os.path.splitext(img_file)[0]}_aug_{i}.jpg')
                    aug_img_pil.save(aug_img_path)

               # print(f"Processed {img_file} from {class_dir}")


# Define the paths
dataset_dir = 'dataset/train_data'
output_dir = 'dataset/augmented_dataset'

# Process the dataset
process_dataset(dataset_dir, output_dir)