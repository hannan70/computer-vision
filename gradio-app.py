import gradio as gr
from PIL import Image
import numpy as np
import cv2

def greet(name, slider_value):
    return f"Hi {name}. Your slider value is {slider_value}"

demo = gr.Interface(
    fn=greet,
    inputs=["textbox", "slider"],
    outputs=["textbox"],
)


def sum_of_total(text_area, number, image):
    number = int(number)
    total = 0
    for i in range(number):
        total += i
    
    return text_area, number, image


def sepia(input_img): 
    image = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    return image

demo3 = gr.Interface(
    fn=sepia,
    inputs=gr.Image(width=500, height=400, label="Upload Image"),
    outputs="image"
)


demo2 = gr.Interface(
    fn=sum_of_total,
    inputs=[
        gr.Textbox(lines=5, placeholder="Enter your prompt") , 
        gr.Slider(0, 100, step=1, label="select a number")],

    outputs=['text']
)

interface_tab = gr.TabbedInterface(
    [demo, demo2, demo3],
    ["Greet", "Demo2", "Demo 3"]
)

interface_tab.launch()