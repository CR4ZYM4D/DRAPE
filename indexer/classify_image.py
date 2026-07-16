# library import
import torch
import os
import sys
import numpy as np
from PIL import Image
from torchvision import transforms

# utility import
from models.attribute_classifier import AttributeClassifier
from logger.logger import logging
from exception.exception import ProjectError

class ImageClassifier:

    """ class for image classifier to transform and label the images """
    
    def __init__(self, checkpoint_path=None):
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AttributeClassifier(pretrained=False).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            logging.info(f" ----- Loading classifier from {checkpoint_path} ----- ")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            logging.warning(" ----- No checkpoint provided or found. Using untrained weights. ----- ")
            
        self.model.eval()

        # image transformer    
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def get_attributes(self, image: Image.Image) -> list[float]:
        """returns 75-dim soft probability vector"""
        try:
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            logits = self.model(input_tensor)
            
            # apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().tolist()
            return probs
        except Exception as e:
            logging.error(f"Error classifying image: {e}")
            raise ProjectError(str(e), sys)

# singleton instance
classifier_instance = None

def get_classifier(checkpoint_path="models/checkpoints/classifier_final.pt"):
    global classifier_instance
    if classifier_instance is None:
        classifier_instance = ImageClassifier(checkpoint_path)
    return classifier_instance
