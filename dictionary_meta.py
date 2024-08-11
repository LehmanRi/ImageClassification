import scipy.io
import numpy as np  # Import numpy

# Load the .mat file
mat_file_path = 'meta.mat'
mat_data = scipy.io.loadmat(mat_file_path)

# Accessing the synsets field
synsets = mat_data['synsets']

# Initialize a dictionary to hold the ILSVRC2010_ID and WNID pairs
id_wnid_dict = {}

# Iterate through the synsets to populate the dictionary
for item in synsets:
    # Convert ILSVRC2010_ID to int and WNID to string if necessary
    ilsvrc_id = int(item[0][0][0]) if isinstance(item[0][0][0], (np.integer, int, float)) else int(item[0][0][0][0])
    wnid = str(item[0][1][0]) if isinstance(item[0][1][0], (str, np.str_)) else str(item[0][1][0][0])

    id_wnid_dict[ilsvrc_id] = wnid

# Display the first few items of the dictionary to verify
print(dict(list(id_wnid_dict.items())[:10]))  # Displaying only the first 10 for brevity
