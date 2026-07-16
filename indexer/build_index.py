# library imports
import os
import sys
import glob
import json
from PIL import Image
from dotenv import load_dotenv
from tqdm import tqdm

# utility imports
from indexer.embed_image import get_embedder
from indexer.classify_image import get_classifier
from db.vector_store import VectorStore
from logger.logger import logging
from exception.exception import ProjectError

# load datastore path
load_dotenv()
DATASTORE_PATH = os.getenv("DATASTORE_PATH", "data/raw/images")


def build_index(image_dir=DATASTORE_PATH):
    try:
        logging.info(" ----- Initializing models for indexing ----- ")
        # load siglip embedder and efficeintnet classfier
        embedder = get_embedder()
        classifier = get_classifier()
        # load vector store
        vstore = VectorStore()
        
        # load images
        image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
        logging.info(f"Found {len(image_files)} images to index.")
        
        batch_size = 32
        ids = []
        embeddings = []
        metadatas = []
        
        # lop throguh images
        for i, img_path in tqdm(enumerate(image_files), total=len(image_files)):
            img_id = os.path.basename(img_path).split(".")[0]
            
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                logging.warning(f" ----- Failed to read image {img_path}: {e} ----- ")
                continue
                
            # embed dense visual vector
            emb = embedder.embed_image(img)
            embeddings.append(emb)
            ids.append(img_id)
            
            # extract soft attribute probability vector (75-dim)
            attr_vec = classifier.get_attributes(img)
            
            # ChromaDB metadatas cannot contain list types, only int/float/str
            # So we must json.dumps the attribute list as a string
            metadatas.append({
                "image_path": img_path,
                "attributes": json.dumps(attr_vec)
            })
            
            # flush batch
            if len(ids) >= batch_size:
                vstore.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
                logging.info(f"Indexed {i+1}/{len(image_files)} images...")
                ids = []
                embeddings = []
                metadatas = []
                
        # flush remaining
        if ids:
            vstore.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
            logging.info(f"Indexed {len(image_files)}/{len(image_files)} images...")
            
        logging.info(" ----- Indexing complete. ----- ")
    
    except Exception as e:
        logging.error(f"Error in index building pipeline: {str(e)}")
        raise ProjectError(str(e), sys)

if __name__ == "__main__":
    build_index()
