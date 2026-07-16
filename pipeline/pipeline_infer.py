# library import
import os
import sys
import mlflow
import dagshub
import torch

# utility import
import subprocess
from dotenv import load_dotenv
from logger.logger import logging
from exception.exception import ProjectError


load_dotenv()
# init dagshub repo
dagshub.init(repo_owner=os.getenv("DAGSHUB_USERNAME"), repo_name='DRAPE', mlflow=True)
# init mlfloiw  treacking
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def run_inference_pipeline():
    try:
        logging.info(" ----- Starting inference pipeline ----- ")
        
        # 1. Pull Model from MLflow
        logging.info(" ----- Fetching Production model from MLflow ----- ")
        model_name = os.getenv("MODEL_NAME")
        client = mlflow.tracking.MlflowClient()
        
        os.makedirs("models/weights", exist_ok=True)
        model_path = "models/weights/attribute_classifier.pth"
        
        try:
            versions = client.search_model_versions(f"name='{model_name}'")
            production_version = next((v for v in versions if v.current_stage == "Production"), None)
            
            if production_version:
                run_id = production_version.run_id
                logging.info(f" ----- Found Production model (Run ID: {run_id}). Downloading ----- ")
                
                # Download model state dict
                # Note: mlflow.pytorch saves the full model or state dict.
                # Since we log it directly, we can load it and save its state_dict locally
                loaded_model = mlflow.pytorch.load_model(f"models:/{model_name}/Production")
                torch.save(loaded_model.state_dict(), model_path)
                logging.info(f" ----- Model successfully cached to {model_path} ----- ")
            else:
                logging.warning(" ----- No Production model found in MLflow. Falling back to local/untrained weights. ----- ")
        except Exception as mlf_err:
            logging.warning(f" ----- MLflow connection error: {mlf_err}. Using existing local weights if available. ----- ")
            
        # Start API
        logging.info(" ----- Starting API Server ----- ")
        subprocess.run([sys.executable, "app.py"], check=True)
        
    except Exception as e:
        logging.error(f"Error in inference pipeline: {e}")
        raise ProjectError(str(e), sys)

if __name__ == "__main__":
    run_inference_pipeline()
