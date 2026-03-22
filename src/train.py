import torch
import torch.nn as nn

from models.model import get_model
from src.dataset import get_loaders
from src.utils import evaluate, train_one_epoch


device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    train_loader, val_loader, classes = get_loaders("data")
    print("Classes:", classes)

    images, labels = next(iter(train_loader))
    print("Batch shape:", images.shape)
    print("Labels:", labels[:5])

    model = get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.0001)
    epochs = 20

    print(type(model))
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Val Acc={acc:.4f}")

    torch.save(model.state_dict(), "weights/model.pth")


if __name__ == "__main__":
    main()
