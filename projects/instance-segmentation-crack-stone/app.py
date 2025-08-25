import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io 


# load model
model = YOLO("best.pt")

# setup streamlit
st.title("🧠 AI-Powered Crack Detection & Segmentation")

uploaded_files  = st.file_uploader("Choose an image...", type=["png", "jpeg", "jpg"], accept_multiple_files=True)


if uploaded_files:
    batch_images = [] 
    # Handle multiple file and convert numpy array
    for upload_file in uploaded_files:
        try:
            image = Image.open(io.BytesIO(upload_file.read()))
            img_array = np.array(image) 
            batch_images.append(img_array)
        except Exception as e:
            st.error(f"Error {upload_file.name}: e")
      

    if batch_images:
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
                    cols[idx].image(result_img, caption=f"Detected Result {uploaded_files[idx].name}" ) 
else:
    st.error("Please upload one or more image")
 

