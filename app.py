import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_my_model():
    return load_model("our_model.h5")

model = load_my_model()

# Set expected image size
IMG_SIZE = (224, 224)

# App UI
st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image and click **Predict** to check if the person is affected.")

# Upload image
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_pil = Image.open(uploaded_file).convert('RGB')
    st.image(image_pil, caption="Uploaded Image", width=200)
    # Predict button
    if st.button("🔍 Predict"):
        # Preprocess image
        img = image_pil.resize(IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize as in your Flask app

        # Predict
        prediction = model.predict(img_array)

        # Interpret result
        if prediction[0][0] > prediction[0][1]:
            st.success("✅ Person is Safe.")
        else:
            st.error("⚠ Person is Affected with Pneumonia.")
