import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd  # Import pandas

# Define constants
IMAGE_SIZE = 299  # InceptionV3 requires input images to be at least 299x299
BATCH_SIZE = 32
NUM_CLASSES = 2  # Number of waste categories: organic and recyclable
EPOCHS = 10

# Directory containing all waste data
data_dir = "waste_data"

# List all image paths and corresponding labels
image_paths = []
labels = []

for category in os.listdir(data_dir):
    category_dir = os.path.join(data_dir, category)
    if os.path.isdir(category_dir):  # Check if it's a directory
        for image_name in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_name)
            image_paths.append(image_path)
            labels.append(category)

# Split data into training and validation sets
train_image_paths, val_image_paths, train_labels, val_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load and augment training data
train_generator = train_datagen.flow_from_dataframe(
    pd.DataFrame({"filename": train_image_paths, "class": train_labels}),
    x_col="filename",
    y_col="class",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load and augment validation data
val_generator = val_datagen.flow_from_dataframe(
    pd.DataFrame({"filename": val_image_paths, "class": val_labels}),
    x_col="filename",
    y_col="class",
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load pre-trained InceptionV3 model
base_model = tf.keras.applications.InceptionV3(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model layers
base_model.trainable = False

# Add classification head
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Save the model
model.save("waste_classification_model_inceptionv3.h5")
