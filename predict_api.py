from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# Initialize Flask application
app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model("waste_classification_model_inceptionv3.h5")

# Define constants
IMAGE_SIZE = 299  # Same as during training
waste_categories = ['Organic Waste', 'Recyclable Waste']

# API route for prediction
@app.route('/predict', methods=['GET'])
def predict():
    # Get image file from request
    file = "test.jpg"
    
    # Load and preprocess the image
    img = image.load_img(file, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.
    
    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    predicted_category = waste_categories[predicted_class]
    
    # Return the predicted waste category
    return jsonify({'predicted_category': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)