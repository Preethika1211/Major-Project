import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("waste_classification_model_inceptionv3.h5")

# Define constants
IMAGE_SIZE = 299  # Same as during training+

# Load and preprocess the test image
img_path = "test.jpg"  # Replace with the path to your test image
img = image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# Define waste categories
waste_categories = ['Organic Waste', 'Recyclable Waste']

# Print the predicted waste category
predicted_category = waste_categories[predicted_class]
print("Predicted waste category:", predicted_category)
