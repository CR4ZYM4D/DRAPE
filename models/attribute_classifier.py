import torch
import torch.nn as nn
import timm
import os
import sys

from config.attribute_schema import TOTAL_DIMS
from logger.logger import logging
from exception.exception import ProjectError

class AttributeClassifier(nn.Module):
    def __init__(self, num_classes=TOTAL_DIMS, pretrained=True):
        super().__init__()
        # load EfficientNet-B0 backbone
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0) # num_classes=0 removes the original classifier
        
        # add 75-dim classification head
        in_features = self.backbone.num_features
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        try:
            features = self.backbone(x)
            logits = self.classifier(features)
            return logits
        except Exception as e:
            logging.error(f"Error in AttributeClassifier forward pass: {str(e)}")
            raise ProjectError(str(e), sys)

if __name__ == "__main__":
    try:
        model = AttributeClassifier()
        dummy_input = torch.randn(2, 3, 224, 224)
        logits = model(dummy_input)
        logging.info(f"Logits shape: {logits.shape}") # Should be (2, 75)
    except Exception as e:
        logging.error(f"Error testing AttributeClassifier: {str(e)}")
        raise ProjectError(str(e), sys)
