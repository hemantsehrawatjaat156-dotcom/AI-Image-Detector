import torch.nn as nn
from torchvision import models


def get_model(pretrained=True):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None

    try:
        model = models.resnet50(weights=weights)
    except (OSError, PermissionError):
        model = models.resnet50(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

