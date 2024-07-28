import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

# Load the TensorFlow Lite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="crop_classifier_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
class_labels = {
    0: 'Maize',
    1: 'Wheat',
}

def prepare_image(image):
    """
    Preprocess the image to the required input format for the model.
    """
    image = image.resize((150, 150))  # Resize to match the input shape required by the model
    image = img_to_array(image)  # Convert image to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image.astype(np.float32)

st.title("Crop Classification")
st.write("Upload an image to classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image file
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Prepare the image for the model
    prepared_image = prepare_image(image)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], prepared_image)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(output_data)
    predicted_class = class_labels.get(predicted_class_index, "Unknown")

    st.write(f"The predicted class of crop is: {predicted_class}")
