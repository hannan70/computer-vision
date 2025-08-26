import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image



# Page Title with Emoji/Icon
st.title("🌿 Plants Classification and Pose Estimation 🤖")

# Sidebar with icon
options = st.sidebar.radio(
    "🔎 Choose Options",
    ['🌱 Classification', '🧍 Pose Estimation']
)

# Load classification model
cls_model = YOLO("plants.pt")

# Load pose estimation model
pose_model = YOLO("pose.pt")



uploaded_file = st.sidebar.file_uploader("Upload a image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    if options == "🌱 Classification":
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with col2:
            with st.spinner("Running Classification Model..."):
                image = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(image)
                results = cls_model(image_np)

                for result in results:

                    probs = result.probs
                    top1_idx = probs.top1
                    top1_conf = probs.top1conf
                    class_name = result.names[top1_idx]
                    st.success(f"🌱 Predicted: **{class_name}** ({top1_conf:.2f})")
                    pred_image = result.plot()
                    st.image(pred_image, caption="Predicted Image")

         
    elif options == "🧍 Pose Estimation":
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        with col2:
            with st.spinner("Running Pose Estimation..."):
                image = Image.open(uploaded_file).convert("RGB")
                image_np = np.array(image)
                results = pose_model(image_np)

                for result in results:
                    pred_image = result.plot()
                    st.image(pred_image, caption="Predicted Image")



