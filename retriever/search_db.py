# library import
import sys
from db.vector_store import VectorStore

# utility import
import logging
from indexer.embed_image import get_embedder
from logger.logger import logging
from exception.exception import ProjectError

class Searcher:

    """ class for searching the chroma vector db based on qery text converted into siglip embeddings"""

    def __init__(self):
        try:
            logging.info(" ----- initializing searcher ----- ")
            self.embedder = get_embedder()
            self.vstore = VectorStore()
        except Exception as e:
            logging.error(f"Failed to initialize searcher: {e}")
            raise ProjectError(str(e), sys)
        
    def search(self, query_text: str, n_results: int = 50):
        try:
            query_emb = self.embedder.embed_text(query_text)
            
            # embed_text returns a list if it's a single query, ensure it's nested for Chroma
            if isinstance(query_emb[0], float):
                query_emb = [query_emb]
                
            results = self.vstore.query(query_embeddings=query_emb, n_results=n_results)
            return results
        except Exception as e:
            logging.error(f"Error executing search query: {e}")
            raise ProjectError(str(e), sys)

# singleton instance
searcher_instance = None

def get_searcher():
    global searcher_instance
    if searcher_instance is None:
        searcher_instance = Searcher()
    return searcher_instance
