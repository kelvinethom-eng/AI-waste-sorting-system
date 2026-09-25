import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from gradcam import make_gradcam_heatmap, overlay_heatmap

st.set_page_config(page_title="Waste Classifier", layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('waste_classifier.h5')

model = load_model()
CLASSES = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

st.title("♻️ AI Waste Sorting System")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(uploaded_file).convert('RGB')
    img_array = tf.expand_dims(tf.keras.preprocessing.image.img_to_array(image.resize((224, 224))), 0)

    with st.spinner("Analyzing..."):
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) 
        st.success(f"**Prediction:** {CLASSES[np.argmax(score)]} ({100 * np.max(score):.1f}%)")

        cam_image = cv2.cvtColor(overlay_heatmap("temp.jpg", make_gradcam_heatmap(img_array, model, 'top_activation')), cv2.COLOR_BGR2RGB)

    c1, c2 = st.columns(2)
    c1.image(image, caption="Original", use_container_width=True)
    c2.image(cam_image, caption="Grad-CAM", use_container_width=True)
