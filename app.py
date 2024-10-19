import gradio as gr
from huggingface_hub import hf_hub_download
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import os
import tensorflow as tf

# model_file = hf_hub_download(
#     repo_id="dilkushsingh/Facial_Emotion_Recognizer",
#     filename="ResNet50_Model.h5",
#     local_dir="./models"
# )
model = load_model('ResNet50_Modelv2.keras')

# Emotion labels dictionary
emotion_labels = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}
index_to_emotion = {v: k for k, v in emotion_labels.items()}

def prepare_image(img_pil):
    """Preprocess the PIL image to fit model's input requirements."""
    # Convert the PIL image to a numpy array with the target size
    img = img_pil.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Convert single image to a batch.
    img_array /= 255.0  # Rescale pixel values to [0,1], as done during training
    return img_array

# Define the Gradio interface
def predict_emotion(image):
    """Predict emotion from an uploaded image."""
    # Preprocess the image
    processed_image = prepare_image(image)
    # Make prediction using the model
    prediction = model.predict(processed_image)
    # Get the emotion label with the highest probability
    predicted_class = np.argmax(prediction, axis=1)
    predicted_emotion = index_to_emotion.get(predicted_class[0], "Unknown Emotion")
    return predicted_emotion

interface = gr.Interface(
    fn=predict_emotion,  # Your prediction function
    inputs=gr.Image(type="pil"),  # Input for uploading an image, directly compatible with PIL images
    outputs="text",  # Output as text displaying the predicted emotion
    title="Facial Emotion Recognizer",
    description="Upload an image and see the predicted emotion."
)

# Launch the Gradio interface
interface.launch()
