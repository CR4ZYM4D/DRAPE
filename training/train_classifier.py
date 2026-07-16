# library import
import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

# utility import
from torch.utils.data import Dataset, DataLoader, random_split
from dotenv import load_dotenv
from models.attribute_classifier import AttributeClassifier
from logger.logger import logging
from exception.exception import ProjectError
from config.attribute_schema import FLAT_ATTRIBUTE_LIST

# load epochs and datastore path
load_dotenv()
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))
DATASTORE_PATH = os.getenv("DATASTORE_PATH", "data/raw/images")
CLASSIFIER_PATH = os.getenv("MODEL_CLASSIFIER_PATH", "models/weights/attribute_classifier.pth")


class FashionDataset(Dataset):

    """ class for training efficient net with the 75 dim multiclass classficiation head and labelling images """

    def __init__(self, annotations_npy, img_dir, transform=None):
        self.labels = np.load(annotations_npy, allow_pickle=True).item()
        self.img_ids = list(self.labels.keys())
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
                
            label = torch.tensor(self.labels[img_id], dtype=torch.float32)
            
            return image, label
        except Exception as e:
            logging.warning(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 224, 224), torch.zeros(len(FLAT_ATTRIBUTE_LIST), dtype=torch.float32)


def compute_pos_weight(labels_path, cap=50.0):

    """
        function to clalculate pos weight to account rarer classes
    """

    labels_dict = np.load(labels_path, allow_pickle=True).item()
    label_matrix = np.stack(list(labels_dict.values())) 

    pos_counts = label_matrix.sum(axis=0)
    total = label_matrix.shape[0]
    neg_counts = total - pos_counts

    pos_weight = neg_counts / np.maximum(pos_counts, 1)
    pos_weight = np.clip(pos_weight, a_min=0.1, a_max=cap)

    rare_classes = [
        (FLAT_ATTRIBUTE_LIST[i], int(pos_counts[i]))
        for i in range(len(pos_counts)) if pos_counts[i] < 5
    ]

    if rare_classes:
        logging.warning(f" ----- WARNING: {len(rare_classes)} classes have <5 positive examples in this dataset: ----- ")
        for name, count in rare_classes:
            logging.warning(f" ----- {name}: {count} positives ----- ")

    return torch.tensor(pos_weight, dtype=torch.float32)


def train(labels_file="data/labels.npy", image_dir=DATASTORE_PATH, batch_size=32, epochs=NUM_EPOCHS, lr=1e-4, val_split=0.1):
    
    try:
        
        logging.info(" ----- Starting training process ----- ")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f" ----- Using device: {device} ----- ")

        if not os.path.exists(labels_file):
            raise FileNotFoundError(f" ----- {labels_file} missing. Run data/label_data.py first. ----- ")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        full_dataset = FashionDataset(labels_file, image_dir, transform=transform)

        val_size = max(1, int(len(full_dataset) * val_split))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = AttributeClassifier(pretrained=True).to(device)

        pos_weight = compute_pos_weight(labels_file).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        logging.info(f" ----- Training on {device} for {epochs} epochs ----- ")

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        os.makedirs("models/weights", exist_ok=True)
        model_save_path = CLASSIFIER_PATH
        torch.save(model.state_dict(), model_save_path)
        logging.info(f" ----- Training complete. Model saved to {model_save_path} ----- ")

    except Exception as e:
        logging.error(f"Error in training pipeline: {str(e)}")
        raise ProjectError(str(e), sys)


if __name__ == "__main__":
    train()