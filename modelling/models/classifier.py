import torch
import torch.nn as nn
from torchvision import models

class Classifier(nn.Module):
    """EfficientNet-based classifier for rice disease classification"""
    
    def __init__(self, num_classes=3, model_name='efficientnet_b0', classifier_units=512, activation_function='relu'):
        super(Classifier, self).__init__()
        

        if model_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(weights='IMAGENET1K_V1')
            num_features = self.backbone.classifier[1].in_features
        else:
            self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
            num_features = self.backbone.classifier[1].in_features
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            # nn.Dropout(0.4),
            nn.Linear(num_features, classifier_units),
            nn.GELU() if activation_function == 'gelu' else nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(classifier_units, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)