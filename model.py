import torch
import torch.nn as nn
from torchvision import models


def baseline_model(num_classes=39):
    """
    Return baseline model (ResNet with fully connected layer replaced).
    """
    model = models.resnet152(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model
