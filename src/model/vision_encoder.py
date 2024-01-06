import torch
import torch.nn as nn

from torchvision.models import resnet34, resnet50, resnet101, resnet152
from torchvision.models.resnet import ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights


class VisionEncoderV1(nn.Module):
    def __init__(self, model_name="resnet34", pretrained=True, out_features=1024):
        super(VisionEncoderV1, self).__init__()

        self.out_features = out_features
        if model_name == "resnet34":
            self.backbone = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        elif model_name == "resnet50":
            self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        elif model_name == "resnet101":
            self.backbone = resnet101(weights=ResNet101_Weights.DEFAULT if pretrained else None)
        elif model_name == "resnet152":
            self.backbone = resnet152(weights=ResNet152_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError("model_name must be in [resnet34, resnet50, resnet101, resnet152]")

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(self.backbone.fc.in_features, out_features)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        x = self.linear(x)
        
        return x