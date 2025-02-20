from flask import Flask, request, render_template, redirect, url_for, session
from PIL import Image
import torch
import torchvision.transforms as transforms
import secrets
import requests
from torchvision import models
import torch.nn as nn
import os

# prepare model
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
response = requests.get(url)
response.raise_for_status()
class_labels = [line.strip() for line in response.text.splitlines()]
model = models.resnet50(pretrained=True)
model.eval()

# prepare app
app = Flask(__name__)
secret = secrets.token_urlsafe(32)
app.secret_key = secret


@app.route("/")
def index():
    # main window
    if os.path.exists(f"./static/{session.get('path', 'NONE')}"):
        os.remove(f"./static/{session.get('path', 'NONE')}")
    return render_template("index.html", text=session.get("text", ""))


@app.route("/upload", methods=["POST"])
def upload_file_and_predict():
    # making a prediction
    if "file" not in request.files:
        session["text"] = "No file part"
        return redirect(url_for("index"))

    file = request.files["file"]
    if file.filename == "":
        session["text"] = "No selected file"
        return redirect(url_for("index"))

    path = f"./static/{file.filename}"
    file.save(path)
    session["path"] = path.split("/")[2]
    print(session["path"])
    if file:
        try:
            image = Image.open(path)

            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            image_transform = transform(image).unsqueeze(0)

            result = model(image_transform)
            _, predicted = torch.max(result, 1)

            probabilities = nn.Softmax(dim=1)(result)
            predicted_class_label = class_labels[predicted.item()]
            predicted_probability = probabilities[0, predicted].item()

            session["text"] = (
                f"Predicted class for your image: {predicted_class_label}, Predicted probability: {predicted_probability:.4f}"
            )
            image.close()
            return redirect(url_for("index"))

        except Exception as e:
            image.close()
            session["text"] = f"Error: {str(e)}"
            return redirect(url_for("index"))


@app.route("/save", methods=["POST", "GET"])
def save_file():
    # saving a file with result of prediction
    if not os.path.exists("./result"):
        os.makedirs("./result")
    try:
        number_of_file = len(
            [
                name
                for name in os.listdir("./result")
                if name.startswith("result_of_detection")
            ]
        )
        with open(f"./result/result_of_detection{number_of_file + 1}.txt", "w") as f:
            f.write(
                session.get("text", "Predicted class for your image: result not found")
                + "\nName of file: "
                + session.get("path", "path for file not found")
            )
        session["text"] = (
            f"the results file is saved in ''./result/result_of_detection{number_of_file + 1}.txt''"
        )
        session["path"] = "NONE"
    except Exception as e:
        session["text"] = f"Error: {str(e)}"
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
