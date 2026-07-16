# library import
import os
import sys

# utility import
from logger.logger import logging
from exception.exception import ProjectError
from data.download_dataset import download_fashionpedia
from data.label_data import process_labels

# dataset building and labelling pipeline 
def run_build_dataset(num_samples: int = None):
    """
        Data-prep-only pipeline: download + label
    """
    try:
    
        num_samples = num_samples or int(os.getenv("NUM_IMAGES", 1000))

        logging.info(" ----- Starting dataset build pipeline (no training) ----- ")

        logging.info(" ----- Downloading dataset ----- ")
        download_fashionpedia(num_samples=num_samples)

        logging.info(" ----- Labeling dataset ----- ")
        process_labels()

        logging.info(" ----- Dataset build complete. (re)build the search index against the current Production model. ----- ")

    except Exception as e:
        logging.error(e)
        raise ProjectError(e, sys)


if __name__ == "__main__":
    run_build_dataset()