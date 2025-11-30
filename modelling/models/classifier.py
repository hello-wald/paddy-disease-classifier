import torch
import torch.nn as nn
from torchvision import models

class Classifier(nn.Module):
    """EfficientNet-based classifier for rice disease classification"""
    def __init__(self, num_classes=3, model_name='efficientnet_b0', classifier_units=512, activation_function='relu', use_pretrained=True):
        super(Classifier, self).__init__()

        if model_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(weights='IMAGENET1K_V1' if use_pretrained else None)
            num_features = self.backbone.classifier[1].in_features
        else:
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1' if use_pretrained else None)
            num_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, classifier_units),
            nn.GELU() if activation_function == 'gelu' else nn.ReLU(),
            nn.Linear(classifier_units, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)