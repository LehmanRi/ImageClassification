import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if not physical_devices:
    print("No GPU found. Exiting.")
    sys.exit(1)
else:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available")

# Set up MirroredStrategy
strategy = tf.distribute.MirroredStrategy()

# Paths to the directories
train_dir = 'dataset/cleaned'

# Image dimensions
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 32
EPOCHS = 90  # Number of epochs as described in the paper


# Function to load preprocessed images
def load_preprocessed_images(folder):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))
    class_dict = {class_name: i for i, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
            img = tf.keras.preprocessing.image.img_to_array(img)
            images.append(img)
            labels.append(class_dict[class_name])

    images = np.array(images, dtype='float32')
    labels = np.array(labels)
    return images, labels, class_dict


# Load the images and labels
images, labels, class_dict = load_preprocessed_images(train_dir)

# Normalize the images
images /= 255.0

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)


def create_alexnet_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(3, 3), strides=2),
        Conv2D(256, (5, 5), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=2),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(384, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(3, 3), strides=2),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


num_classes = len(class_dict)

with strategy.scope():
    model = create_alexnet_model((IMG_HEIGHT, IMG_WIDTH, 3), num_classes)
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Assuming X_train, y_train, X_val, y_val are your training and validation data
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Data augmentation
datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    rotation_range=20
)

# Train the model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(x_val, y_val),
    steps_per_epoch=len(x_train) // BATCH_SIZE,
    epochs=EPOCHS
)

# Save the model
model.save('alexnet_model.h5')

print("Model training completed and saved as 'alexnet_model.h5'")
