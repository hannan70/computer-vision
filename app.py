from flask import Flask, request, jsonify, render_template, send_file 
import numpy as np
import cv2
import base64
from PIL import Image
import io 
from io import BytesIO
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as transforms
import torch


app = Flask(__name__) 


# LetNet Architecture
class LeNet5_MNIST_Sequential(nn.Module):
    def __init__(self):
        super(LeNet5_MNIST_Sequential, self).__init__()

        # Convolution + ReLU + Pool Layers
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), # 28*28
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2), # 14x14

            nn.Conv2d(6, 16, kernel_size=5, stride=1),  # 10x10
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2) # 5*5
        )

        # Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120), # 16*5*5 = 400
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.features(x) 
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet5_MNIST_Sequential().to(device)
model.load_state_dict(torch.load("./image-algorithm/lenet5_mnist.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read the image file
        img = Image.open(file.stream).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28 for MNIST
        
        # Convert to numpy array and apply transformations
        img_array = np.array(img)
        image_tensor = transform(img_array)
 
        with torch.no_grad():
            image = image_tensor.to(device).unsqueeze(0) # Add batch dimension
            output = model(image)
            _, predicted = torch.max(output, 1)
            
        return jsonify({"prediction": int(predicted.item())})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

  

# Preview API (see uploaded image in browser)
@app.route("/preview", methods=["POST"])
def preview():
    if "file" not in request.files:
        return "No file provided", 400

    file = request.files["file"]

    # Read image
    image = Image.open(BytesIO(file.read())).convert("L")

    # Convert PIL image → BytesIO
    img_io = BytesIO()
    image.save(img_io, "PNG")
    img_io.seek(0)

    # Send back as response (so browser shows it)
    return send_file(img_io, mimetype="image/png")



if __name__ == "__main__":
    app.run(debug=True)
