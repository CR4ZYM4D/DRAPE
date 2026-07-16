import os
import sys
import json
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger.logger import logging
from exception.exception import ProjectError

load_dotenv()
# get datastore directory path
DATASTORE_PATH = os.getenv("DATASTORE_PATH", "data/raw/images")

def download_fashionpedia(num_samples: int = 1000, output_dir: str = os.path.dirname(DATASTORE_PATH)):
    
    try:
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        
        logging.info(f" ----- Downloading Fashionpedia subset ({num_samples} samples) -----")
        
        # load fashionpedia dataset
        dataset = load_dataset("detection-datasets/fashionpedia", split="train", streaming=True)
        
        # declare annotations dict
        annotations = {}
        
        # loop through dataset
        for idx, item in enumerate(dataset):

            if idx >= num_samples:
                break

            # get image and image ID    
            image = item["image"]
            image_id = str(item["image_id"])
            
            # Save image
            image_path = os.path.join(output_dir, "images", f"{image_id}.jpg")
            image.save(image_path)
            
            # Get image bounding box and labels (types of clothes present in image and their bounding box) both prsent in objects dict
            # Fashionpedia typically provides bbox as [xmin, ymin, xmax, ymax]
            objects = item.get("objects", {})
            boxes = objects.get("bbox", [])
            categories = objects.get("category", [])
            
            # annotate image with bounding box and catergories like shirt etc.
            annotations[image_id] = {
                "image_path": image_path,
                "boxes": boxes,
                "categories": categories
            }
            
            if (idx + 1) % 200 == 0:
                logging.info(f"Processed {idx + 1} images.")
    
        # save annotations for tensor building
        annotations_path = os.path.join(output_dir, "annotations.json")
        
        with open(annotations_path, "w") as f:
            json.dump(annotations, f, indent=2)
            
        logging.info(f" ----- Downloaded {idx} images and saved annotations to {annotations_path} ----- ")
    
    except Exception as e:
        logging.error(e)
        raise ProjectError(e, sys)

if __name__ == "__main__":
    download_fashionpedia()
