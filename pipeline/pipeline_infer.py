# library import
import sys

# utility import
import subprocess
from logger.logger import logging
from exception.exception import ProjectError


def run_inference_pipeline():

    try:        
        # Start API
        logging.info(" ----- Starting API Server ----- ")
        subprocess.run([sys.executable, "app.py"], check=True)
        
    except Exception as e:
        logging.error(f"Error in inference pipeline: {e}")
        raise ProjectError(str(e), sys)

if __name__ == "__main__":
    run_inference_pipeline()
