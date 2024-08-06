import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Paths to directories and model
model_path = 'alexnet_model.h5'
val_dir = 'dataset/val'  # Replace with actual validation directory path
train_dir = 'dataset/cleaned'  # Replace with the actual training directory path

# Load the trained model
model = tf.keras.models.load_model(model_path)


# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# Function to predict the class of an image
def predict_image_class(model, image_path, class_dict):
    # Preprocess the image
    img_array = preprocess_image(image_path)

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    print("predicted_class_index", predicted_class_index)
    # Map the index to the class label
    class_labels = {v: k for k, v in class_dict.items()}
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label


# Create the class dictionary from the training directory
class_names = sorted(os.listdir(train_dir))
class_dict = {class_name: i for i, class_name in enumerate(class_names)}
print(class_dict)

# Example: Classify an image from the validation set
# Provide the path to an image you want to classify
image_path = 'dataset/val/ILSVRC2010_val_00030137.JPEG'  # Replace with actual image path
actual_class_label = os.path.basename(os.path.dirname(image_path))  # The actual class from the folder name

# Predict the class
predicted_class_label = predict_image_class(model, image_path, class_dict)

print(f"Actual class: {actual_class_label}")
print(f"Predicted class: {predicted_class_label}")

# Check if the prediction is correct
is_correct = actual_class_label == predicted_class_label
print(f"Correctly classified: {is_correct}")
