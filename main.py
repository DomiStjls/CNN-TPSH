from flask import Flask, request, render_template, redirect, url_for, session
from PIL import Image
import torch
import torchvision.transforms as transforms
import secrets
import requests
from torchvision import models
import torch.nn as nn
import os

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(url)
response.raise_for_status()
class_labels = [line.strip() for line in response.text.splitlines()]
model = models.resnet50(pretrained=True) 
model.eval()  
app = Flask(__name__)


secret = secrets.token_urlsafe(32)
app.secret_key = secret


@app.route("/")
def index():
    if os.path.exists("./static/image.jpg"):
        os.remove("./static/image.jpg")
    return render_template("index.html", text=session.get("text", ""))


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]
    path = "./static/image.jpg"
    file.save(path)
    if file.filename == "":
        return "No selected file"

    if file:
        try:
            image = Image.open(path)
            # img = v2.DecodeImage(image)

            transform = transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            ])
            image_transform = transform(image).unsqueeze(0)

            result = model(image_transform)
            _, predicted = torch.max(result, 1)
            
            probabilities = nn.Softmax(dim=1)(result)
            predicted_class_label = class_labels[predicted.item()]
            predicted_probability = probabilities[0, predicted].item()

            session["text"] = (
                f"Predicted class: {predicted_class_label}, Probability: {predicted_probability:.4f}"
            )
            image.close()
            return redirect(url_for("index"))

        except Exception as e:
            image.close()
            session["text"] = f"Error: {str(e)}"
            return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
