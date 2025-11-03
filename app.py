import streamlit as st                # To create the simple web interface
import tensorflow as tf               # To load and use the pre-trained AI model
import numpy as np                    # To handle image data as arrays
from PIL import Image                 # To open uploaded images
import pandas as pd

st.set_page_config(page_title="AI Image Classifier", page_icon="🤖", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 Simple AI Image Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Upload an image of a dog, cat, car, bicycle, or bottle to identify it!</p>", unsafe_allow_html=True)
st.divider()

model = tf.keras.applications.MobileNetV2(weights="imagenet")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Make prediction
    predictions = model.predict(x)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    # Show top predictions
    st.subheader("🔍 Top Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded):
        st.write(f"{i+1}. **{label.title()}** — {score*100:.2f}%")

    # SIMPLE CLASSIFICATION LOGIC
    labels = [label.lower() for (_, label, _) in decoded]

    dog_keywords = ["dog", "hound", "retriever", "poodle", "terrier", "bulldog", "shepherd"]
    cat_keywords = ["cat", "kitten", "siamese", "tabby", "persian"]
    car_keywords = ["car", "convertible", "minivan", "cab"]
    bicycle_keywords = ["bicycle", "bike"]
    bottle_keywords = ["bottle", "water bottle", "beer bottle"]
    fruit_keywords=['banana','Grape','Orange','Pineapple','Pomegranate','Strawberry','Watermelon']

    found_label = None
    confidence = 0

    for (_, label, score) in decoded:
        lower_label = label.lower()
        if any(k in lower_label for k in cat_keywords):
            found_label = "CAT"; confidence = score * 100; break
        elif any(k in lower_label for k in dog_keywords):
            found_label = "DOG"; confidence = score * 100; break
        elif any(k in lower_label for k in car_keywords):
            found_label = "CAR"; confidence = score * 100; break
        elif any(k in lower_label for k in bicycle_keywords):
            found_label = "BICYCLE"; confidence = score * 100; break
        elif any(k in lower_label for k in bottle_keywords):
            found_label = "BOTTLE"; confidence = score * 100; break
        elif any(k in lower_label for k in fruit_keywords):
            found_label="FRUIT"; confidence=score*100; break
    
    data = {"Label": [label.title() for (_, label, _) in decoded],"Confidence": [score*100 for (_, label, score) in decoded]}
    st.bar_chart(pd.DataFrame(data).set_index("Label"))

    # DISPLAY FINAL RESULT
    if found_label:
        st.success(f"The image looks like a **{found_label}** — Confidence: {confidence:.2f}%")
        st.progress(int(confidence))
    else:
        st.warning("Not sure — maybe not one of the supported objects (dog, cat, car, bicycle, bottle, fruits).")

    with st.expander("**i** About this App"):
        st.write("""This is a simple AI-powered image classifier built using **TensorFlow** and **Streamlit**.
        It uses a pre-trained **MobileNetV2** model (trained on ImageNet-1K) to identify common objects like dogs, cats, cars, bicycles, bottles, and some fruits.
        """)

