import gradio as gr
from PIL import Image
import numpy as np
from ultralytics import YOLO

model = YOLO("./model/rock-paper.pt")

def predict(img): 
    results = model.predict(source=img)
    print(results)
    annotated_frame = results[0].plot()

    annotated_img = annotated_frame[:, :, ::-1]

    return annotated_img


app = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Rock ✊, Paper ✋, Scissors ✌️"),
    outputs=gr.Image(type="numpy", label="Prediction"),
    title="🪨📄✂️ Rock-Paper-Scissors Detection",
    description="Upload Image!"
)

app.launch()
