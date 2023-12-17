import torch
import torch.nn as nn

from torchvision.models import resnet34, resnet50, resnet101, resnet152


class VisionEncoderV1(nn.Module):
    def __init__(self, model_name="resnet34", pretrained=True, out_features=1024):
        super(VisionEncoderV1, self).__init__()

        self.out_features = out_features
        if model_name == "resnet34":
            self.backbone = resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            self.backbone = resnet50(pretrained=pretrained)
        elif model_name == "resnet101":
            self.backbone = resnet101(pretrained=pretrained)
        elif model_name == "resnet152":
            self.backbone = resnet152(pretrained=pretrained)
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