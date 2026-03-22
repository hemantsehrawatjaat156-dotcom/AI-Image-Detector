from flask import Flask, render_template, request
import torch
from torchvision import transforms
from PIL import Image
from models.model import get_model
import gradio as gr

app = Flask(__name__)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_model(pretrained=False).to(device)
model.load_state_dict(torch.load("weights/model.pth", map_location=device))
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

classes = ['fake', 'real']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if file is None or file.filename == "":
        return render_template("index.html", prediction="No image selected")
    img = Image.open(file).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).item()

    return render_template("index.html", prediction=classes[pred])

if __name__ == "__main__":
    app.run(debug=True)


def greet(name):
    return "Hello " + name + "!!"
    

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo.launch()

