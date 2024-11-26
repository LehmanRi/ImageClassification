import numpy as np

# Define mapping from WNIDs to class indices
class_to_wnid = {
    'n01484850': 0, 'n01496331': 1, 'n01664065': 2, 'n01675722': 3,
    'n01685808': 4, 'n01695060': 5, 'n01728920': 6, 'n01770081': 7
}

# Initialize an 8x8 matrix to store classification counts
confusion_matrix = np.zeros((8, 8), dtype=int)

# Read classification results from file
file_path = 'classification_results.txt'
with open(file_path, 'r') as file:
    for line in file:
        # Skip empty lines
        if line.strip() == "":
            continue

        # Parse the line to extract true WNID and predicted WNID
        parts = line.split(', ')

        # Check if line format is as expected (has at least 3 parts for true WNID, predicted WNID, and confidence)
        if len(parts) < 3:
            print(f"Skipping line due to unexpected format: {line}")
            continue

        # Extract WNID values from parts
        try:
            true_wnid = parts[1].split(': ')[1].strip()
            pred_wnid = parts[3].split(': ')[1].strip()
        except IndexError as e:
            print(f"Error parsing line: {line}")
            continue

        # Get corresponding indices if the WNIDs are in the mapping
        if true_wnid in class_to_wnid and pred_wnid in class_to_wnid:
            true_index = class_to_wnid[true_wnid]
            pred_index = class_to_wnid[pred_wnid]
            confusion_matrix[true_index, pred_index] += 1
        else:
            print(f"WNID not found in mapping: True WNID: {true_wnid}, Predicted WNID: {pred_wnid}")

# Save the confusion matrix to a file
output_file_path = 'confusion_matrix.txt'
with open(output_file_path, 'w') as output_file:
    output_file.write("Confusion Matrix:\n")
    for row in confusion_matrix:
        output_file.write(" ".join(map(str, row)) + "\n")

print(f"Confusion matrix saved to {output_file_path}")
