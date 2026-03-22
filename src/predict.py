from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

model = get_model(pretrained=False)
model.load_state_dict(torch.load("weights/model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1).item()

    return "fake" if pred == 0 else "real"

if __name__ == "__main__":
    print(predict("data/val/real/yash.png"))
