# library import
import sys
import numpy as np

# utility import
from indexer.embed_image import get_embedder
from config.attribute_schema import FLAT_ATTRIBUTE_LIST, TOTAL_DIMS
from logger.logger import logging
from exception.exception import ProjectError

class QueryParser:
    def __init__(self):
        try:
            self.embedder = get_embedder()
            logging.info(" ----- Pre-computing SigLIP embeddings for all schema attributes ----- ")
            
            formatted_attrs = [f"a photo containing {attr}" for attr in FLAT_ATTRIBUTE_LIST]
            self.attribute_embeddings = np.array(self.embedder.embed_text(formatted_attrs))
        except Exception as e:
            logging.error(f"Failed to initialize QueryParser: {e}")
            raise ProjectError(str(e), sys)
        
    def parse_query(self, query_text: str, top_k: int = 12) -> np.ndarray:
        """
        Parses a raw text query into a soft attribute vector using SigLIP zero-shot matching.
        """
        try:
            query_attr_vec = np.zeros(TOTAL_DIMS, dtype=np.float32)
            
            query_emb = np.array(self.embedder.embed_text(query_text)).flatten()
            similarities = np.dot(self.attribute_embeddings, query_emb.T).flatten()
            similarities = np.maximum(similarities, 0.0)
        
            # keep only the top_k highest-scoring attributes, zero out the rest
            top_indices = np.argsort(similarities)[::-1][top_k:]
            similarities[top_indices] = 0.0
        
            if similarities.max() > 0:
                similarities = similarities / similarities.max()
                
            query_attr_vec += similarities
            
            logging.info(f" ----- Parsed query '{query_text}' with top concepts: {[(FLAT_ATTRIBUTE_LIST[i], float(query_attr_vec[i])) for i in np.argsort(query_attr_vec)[-3:][::-1] if query_attr_vec[i] > 0]} ----- ")
            return query_attr_vec
        except Exception as e:
            logging.error(f"Error parsing query: {e}")
            raise ProjectError(str(e), sys)

# singleton instance
parser_instance = None

def get_query_parser():
    global parser_instance
    if parser_instance is None:
        parser_instance = QueryParser()
    return parser_instance

if __name__ == "__main__":
    parser = get_query_parser()
    query = "A red tie and a white shirt in a formal setting"
    vec = parser.parse_query(query)
    
    print(f"Top matched attributes for query: '{query}'")
    top_indices = vec.argsort()[::-1][:10]
    for idx in top_indices:
        print(f"{FLAT_ATTRIBUTE_LIST[idx]}: {vec[idx]:.4f}")
