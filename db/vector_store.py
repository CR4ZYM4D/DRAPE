# library imports
import os
import sys
import chromadb

# utility imports
from dotenv import load_dotenv
from logger.logger import logging
from exception.exception import ProjectError

load_dotenv()
# load local db path
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
CHROMA_HOST = os.getenv("CHROMA_HOST", None)
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))


class VectorStore:
    """ class for storing the combined dense (attribute schema) vector and siglip embedding vectors for each image in vectorDB"""

    def __init__(self, collection_name: str = "drape_images"):
    
        self.collection_name = collection_name
        if CHROMA_HOST:
            logging.info(f" ----- connecting to remote Chroma server at {CHROMA_HOST}:{CHROMA_PORT} ----- ")
            self.client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        else:
            logging.info(f" ----- using local Chroma PersistentClient at {CHROMA_PATH} ----- ")
            self.client = chromadb.PersistentClient(path=CHROMA_PATH)
            
        self.collection = self.init_collection()

    def init_collection(self):
        
        """Initializes the ChromaDB collection."""
        
        try:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"} # cosine similarity for SigLIP dense embeddings default is L2
            )
        except Exception as e:
            logging.error(f" ----- Error initializing ChromaDB collection: {e} ----- ")
            raise ProjectError(str(e), sys)

    def upsert(self, ids: list[str], embeddings: list[list[float]], metadatas: list[dict]):
        """uploads vectors and metadata into the collection."""
        try:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            logging.error(f"Error upserting to ChromaDB: {e}")
            raise ProjectError(str(e), sys)
        
    def query(self, query_embeddings: list[list[float]], n_results: int = 50):
        """queries the collection for the closest matches."""
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results
            )
            return results
        except Exception as e:
            logging.error(f"Error querying ChromaDB: {e}")
            raise ProjectError(str(e), sys)
