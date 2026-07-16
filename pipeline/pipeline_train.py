#library import
import os
import sys
import mlflow
import dagshub
import mlflow.pytorch
import torch
import numpy as np

# utility import
from dotenv import load_dotenv
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from training.train_classifier import compute_pos_weight, FashionDataset
from models.attribute_classifier import AttributeClassifier
from logger.logger import logging
from exception.exception import ProjectError


load_dotenv()
# init dagshub repo
dagshub.init(repo_owner=os.getenv("DAGSHUB_USERNAME"), repo_name='DRAPE', mlflow=True)

# init mlflow tracking
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))
DATASTORE_PATH = os.getenv("DATASTORE_PATH", "data/raw/images")

CHECKPOINT_PATH = os.getenv("MODEL_CLASSIFIER_PATH","models/checkpoints/classifier_final.pt")

VAL_SPLIT_SEED = 42

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("DRAPE_Attribute_Classifier")


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return precision, recall, f1


def run_pipeline():
    """
    Training-only pipeline: assumes data prep (download + label) has already
    been run via pipeline_build_dataset.py. Growing the dataset and retraining
    the classifier are deliberately separate steps — see pipeline_build_dataset.py.
    """
    try:
        logging.info(" ----- Starting automated training pipeline ----- ")

        logging.info(" -----  Verifying labeled dataset exists ----- ")
        labels_file = "data/labels.npy"

        if not os.path.exists(labels_file):
            raise ProjectError(
                f"{labels_file} missing. Run pipeline_build_dataset.py first.", sys
            )
            

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        full_dataset = FashionDataset(labels_file, DATASTORE_PATH, transform=transform)
        val_size = max(1, int(len(full_dataset) * 0.1))
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(VAL_SPLIT_SEED),
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AttributeClassifier(pretrained=True).to(device)
        pos_weight = compute_pos_weight(labels_file).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        logging.info(" -----  MLflow Training ----- ")
        with mlflow.start_run() as run:
            mlflow.log_param("epochs", NUM_EPOCHS)
            mlflow.log_param("batch_size", 32)
            mlflow.log_param("learning_rate", 1e-4)
            mlflow.log_param("val_split_seed", VAL_SPLIT_SEED)

            for epoch in range(NUM_EPOCHS):
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                avg_loss = running_loss / len(train_loader)
                logging.info(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
                mlflow.log_metric("train_loss", avg_loss, step=epoch)

            logging.info(" ----- Evaluating Model ----- ")
            precision, recall, f1 = evaluate_model(model, val_loader, device)

            logging.info(f" ----- Eval Results -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f } ----- ")
            mlflow.log_metric("val_precision", precision)
            mlflow.log_metric("val_recall", recall)
            mlflow.log_metric("val_f1", f1)

            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)

            logging.info(" -----  Model Registry Check ----- ")
            model_name = "DRAPE_Classifier"
            client = mlflow.tracking.MlflowClient()

            best_f1 = 0.0
            try:
                versions = client.search_model_versions(f"name='{model_name}'")
                for v in versions:
                    if v.current_stage == "Production":
                        run_id = v.run_id
                        metrics = client.get_run(run_id).data.metrics
                        best_f1 = metrics.get("val_f1", 0.0)
                        break
            except Exception as registry_err:
                logging.warning(f"No existing registered model found or error accessing registry: {registry_err}")

            if f1 > best_f1:
                logging.info(f"New model outperforms previous F1 ({f1:.4f} > {best_f1:.4f}). Registering!")

                model_info = mlflow.pytorch.log_model(
                    model, "model", registered_model_name=model_name, serialization_format="pickle"
                )
                new_version = model_info.registered_model_version

                client.transition_model_version_stage(
                    name=model_name,
                    version=new_version,
                    stage="Production"
                )
            else:
                logging.info(f" ----- New model did not outperform (F1: {f1:.4f} <= {best_f1:.4f}). Logging but not registering as Production. ----- ")

        logging.info(" ----- Training pipeline complete. ----- ")

    except Exception as e:
        logging.error(f"Error in pipeline_train: {e}")
        raise ProjectError(str(e), sys)


if __name__ == "__main__":
    run_pipeline()