# model.py
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetDR(nn.Module):
    def __init__(self, num_classes=5):
        super(MobileNetDR, self).__init__()
        
        # Load MobileNetV3 with latest ImageNet weights
        weights = MobileNet_V3_Small_Weights.DEFAULT
        self.model = mobilenet_v3_small(weights=weights)

        # Replace the final classifier layer
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)