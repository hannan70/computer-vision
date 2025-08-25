import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io 
import cv2
import tempfile
import os


# load model
model = YOLO("best.pt")

# Inject custom CSS for sidebar width
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            min-width: 400px;
            max-width: 400px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# setup streamlit
st.title("🧠 AI-Powered Pothole Detection & Segmentation")


options = st.sidebar.radio(
    "Choose file type",
    ("Image", "Video")
)
batch_images = []
uploaded_file  = None

if options == "Image":
    uploaded_file  = st.sidebar.file_uploader("Choose an image...", type=["png", "jpeg", "jpg"], accept_multiple_files=True)
    if uploaded_file : 
        # Handle multiple image and convert numpy array
        for file in uploaded_file:
            try:
                image = Image.open(io.BytesIO(file.read()))
                img_array = np.array(image)
                batch_images.append(img_array)
            except Exception as e:
                st.error(f"Error {file.name}: e")

elif options == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4"])


if uploaded_file and st.sidebar.button("Predict"):
        
        # Handle image
        if options == "Image": 
            with st.spinner("Running Crack Detection..."):
                results = model(batch_images)
                if len(results) == 1:
                    col_per_row = 1
                else:
                    col_per_row = 3

                for i in range(0, len(results), col_per_row):
                    cols = st.columns(col_per_row)
                    for idx, result in enumerate(results[i: i +col_per_row]):
                        result_img = result.plot()
                        cols[idx].image(result_img, caption=f"Detected Result {uploaded_file[idx].name}" ) 

        # handle video
        elif options == "Video":
            try:
                # Step 1: Save uploaded file to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.close()  # important

                # Step 2: Open temp file with OpenCV
                video_cap = cv2.VideoCapture(tfile.name)
                st_frame = st.empty()

                while video_cap.isOpened():
                    success, frame = video_cap.read()
                    if not success:
                        break

                    # Resize frame
                    frame = cv2.resize(frame, (720, int(720 * (9/16))))

                    # Step 3: Predict using YOLO
                    results = model.predict(frame)
                    result_frame = results[0].plot()

                    # Step 4: Show in Streamlit
                    st_frame.image(result_frame, channels="BGR")

                video_cap.release()

            except Exception as e:
                st.sidebar.error("Error Processing Video: " + str(e))