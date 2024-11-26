import os
import shutil
import scipy.io
import numpy as np

# Load the .mat file
mat_file_path = 'meta.mat'
mat_data = scipy.io.loadmat(mat_file_path)

# Access the synsets field
synsets = mat_data['synsets']

# Initialize a dictionary to hold the ILSVRC2010_ID and WNID pairs
id_wnid_dict = {}
for item in synsets:
    ilsvrc_id = int(item[0][0][0]) if isinstance(item[0][0][0], (np.integer, int, float)) else int(item[0][0][0][0])
    wnid = str(item[0][1][0]) if isinstance(item[0][1][0], (str, np.str_)) else str(item[0][1][0][0])
    id_wnid_dict[ilsvrc_id] = wnid

# Display the first few items of the dictionary to verify
print(dict(list(id_wnid_dict.items())[:10]))

# Define the path to the text file and the images folder
ground_truth_file = 'ILSVRC2010_test_ground_truth.txt'
images_folder = 'ILSVRC2010_images_test/test'
output_folder = 'test_images'

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Define the target WNIDs
target_wnids = {'n01484850', 'n01496331', 'n01664065', 'n01675722', 'n01685808', 'n01695060', 'n01728920', 'n01770081'}

# Read the ground truth file and process each image
with open(ground_truth_file, 'r') as f:
    for i, line in enumerate(f):
        ilsvrc_id = int(line.strip())  # Get the ILSVRC ID for the current image
        wnid = id_wnid_dict.get(ilsvrc_id)  # Look up the corresponding WNID

        if wnid in target_wnids:  # Check if the WNID is in the target list
            # Define source and destination paths for the image
            image_name = f"ILSVRC2010_test_{i+1:08d}.JPEG"  # Use the correct naming format
            src_path = os.path.join(images_folder, image_name)
            dst_path = os.path.join(output_folder, f"{wnid}_{i+1:08d}.JPEG")  # Save with WNID and image index for uniqueness

            # Copy the image to the destination with the WNID as the filename
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                print(f"Image {src_path} not found.")

print("Images have been copied successfully.")
