import os
import shutil
import numpy as np
import scipy.io

# Define paths
val_images_folder = 'dataset/val_data'
output_folder = 'dataset/divided_val'  # Where to create subfolders and copy images
labels_file = 'dataset/ILSVRC2010_validation_ground_truth.txt'

# Load the .mat file
mat_file_path = 'meta.mat'
mat_data = scipy.io.loadmat(mat_file_path)

# Accessing the synsets field
synsets = mat_data['synsets']

# Initialize dictionaries to hold the mappings
id_to_wnid_dict = {}

# Populate the dictionary mapping from ILSVRC ID to WNID
for item in synsets:
    ilsvrc_id = int(item[0][0][0])  # ILSVRC2010_ID
    wnid = str(item[0][1][0])  # WNID
    id_to_wnid_dict[ilsvrc_id] = wnid

# Read labels file
labels = np.loadtxt(labels_file, dtype=int)

# Track the WNID folders that should be created
wnid_to_images = {}

# Categorize images by their WNID
for idx, label_id in enumerate(labels):
    wnid = id_to_wnid_dict[label_id]
    image_name = f"ILSVRC2010_val_{idx + 1:08d}.JPEG"
    source_path = os.path.join(val_images_folder, image_name)

    if os.path.exists(source_path):
        if wnid not in wnid_to_images:
            wnid_to_images[wnid] = []
        wnid_to_images[wnid].append(image_name)

# Create the folders and copy the images
for wnid, images in wnid_to_images.items():
    wnid_folder = os.path.join(output_folder, wnid)
    os.makedirs(wnid_folder, exist_ok=True)

    for image_name in images:
        source_path = os.path.join(val_images_folder, image_name)
        destination_path = os.path.join(wnid_folder, image_name)
        shutil.copy(source_path, destination_path)

print("Images have been categorized and copied without creating empty folders.")
