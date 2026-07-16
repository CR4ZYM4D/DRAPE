# library import
import sys
import os

# utility import
from logger.logger import logging
from exception.exception import ProjectError
from indexer.build_index import build_index

def run_indexing_pipeline():
    
    try:
        images_folder = os.getenv("DATASTORE_PATH")

        if not os.path.exists(images_folder):
            raise Exception(f" ----- {images_folder} missing. Run pipeline_build_dataset.py first ----- ")
        
        logging.info(" ----- Starting standalone indexing pipeline ----- ")
        build_index()
        logging.info(" ----- Indexing pipeline completed successfully ----- ")
    except Exception as e:
        logging.error(f" ----- Error in indexing pipeline: {e} ----- ")
        raise ProjectError(str(e), sys)

if __name__ == "__main__":
    run_indexing_pipeline()
