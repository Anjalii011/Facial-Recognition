import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tempfile
from PIL import Image

# Load the pre-trained model
model = load_model("model.h5")  # Replace with your model file

# Emotion class labels
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to preprocess and predict emotion
def detect_emotion(img_path):
    try:
        # Open the image and convert it to grayscale
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        img = img.resize((48, 48))  # Resize to model input size
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

        # Predict emotion
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = round(prediction[0][predicted_index] * 100, 2)

        return predicted_class, confidence, img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None, None

# Streamlit App
st.title("🧠 Emotion Detection App")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    try:
        # Save uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file.write(uploaded_file.read())
        temp_file.close()

        # Perform emotion detection
        predicted_emotion, confidence, img = detect_emotion(temp_file.name)

        if predicted_emotion:
            # Display results
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"### Predicted Emotion: **{predicted_emotion}**")
            st.markdown(f"### Confidence: **{confidence}%**")

            # Show the image with prediction
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Predicted Emotion: {predicted_emotion} ({confidence}%)')
            ax.axis('off')
            st.pyplot(fig)
    except Exception as e:
        st.error(f"Error: {e}")
