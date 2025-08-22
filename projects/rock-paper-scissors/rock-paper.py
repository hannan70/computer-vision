import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

model = YOLO("./model/best.pt")


file_upload = st.file_uploader("Upload you file")
print(file_upload)

if file_upload is not None:
    img = Image.open(file_upload).convert("RGB")
    img_array = np.array(img)

    results = model.predict(source=img_array, conf=0.25, save=True)
    print(results)
    for res in results:
        st.write(f"Detected object {len(res.boxes)}")
        print(len(res.boxes))
        im_array = res.plot()
        st.image(im_array, caption="Detected result", use_container_width=True )


