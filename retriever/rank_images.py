# library import 
import os
import sys
import json
import numpy as np

# utility import
from dotenv import load_dotenv
from retriever.query_parser import get_query_parser
from retriever.search_db import get_searcher
from logger.logger import logging
from exception.exception import ProjectError

# load max count of images upto which all images are queried for similarity
load_dotenv()
EXHAUSTIVE_THRESHOLD_ENV = int(os.getenv("EXHAUSTIVE_THRESHOLD", "5000"))

def cosine_overlap(a, b, eps=1e-8):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + eps))


class Ranker:

    """class for ranking the images based on similarity with the prompt/query by the help of using cos similarity on the embedded attribute and siglip combined vector"""
    def __init__(self, lambda_weight=0.25):
        try:
            logging.info(" ----- Initializing image ranker class ----- ")
            self.parser = get_query_parser()
            self.searcher = get_searcher()
            self.lambda_weight = lambda_weight
            self.EXHAUSTIVE_THRESHOLD = EXHAUSTIVE_THRESHOLD_ENV
        except Exception as e:
            logging.error(f"Failed to initialize Ranker: {e}")
            raise ProjectError(str(e), sys)


    def search_and_rank(self, query_text: str, top_k=5):
        try:
            # parse query to soft attribute vector
            query_attr_vec = self.parser.parse_query(query_text)
        
            collection_size = self.searcher.vstore.collection.count()
        
            if collection_size <= self.EXHAUSTIVE_THRESHOLD:
                # exhaustive scoring score every indexed image directly
                all_data = self.searcher.vstore.collection.get(include=["embeddings", "metadatas"])
                ids = all_data["ids"]
                embeddings = np.array(all_data["embeddings"])
                metadatas = all_data["metadatas"]
        
                if len(ids) == 0:
                    return []
        
                query_emb = np.array(self.searcher.embedder.embed_text(query_text))
                dense_sims = embeddings @ query_emb
        
            else:
                # Large-scale path Chroma HNSW top-K dense fetch, then rerank
                n_results = min(500, collection_size // 20)
                raw_results = self.searcher.search(query_text, n_results=n_results)
                if not raw_results['ids'] or not raw_results['ids'][0]:
                    return []
                ids = raw_results['ids'][0]
                distances = raw_results['distances'][0]
                metadatas = raw_results['metadatas'][0]
                dense_sims = 1.0 - np.array(distances)
        
            # Reranking 
            scored_candidates = []
            for i in range(len(ids)):
                img_id = ids[i]
                img_path = metadatas[i]['image_path']
                attr_str = metadatas[i].get('attributes', '[]')
                try:
                    candidate_attr_vec = np.array(json.loads(attr_str))
                    if len(candidate_attr_vec) == 0:
                        candidate_attr_vec = np.zeros_like(query_attr_vec)
                except json.JSONDecodeError:
                    candidate_attr_vec = np.zeros_like(query_attr_vec)
        
                attribute_overlap = cosine_overlap(query_attr_vec, candidate_attr_vec)
                dense_score = float(dense_sims[i])
                final_score = dense_score + self.lambda_weight * attribute_overlap
        
                scored_candidates.append({
                    "id": img_id,
                    "image_path": img_path,
                    "score": float(final_score),
                    "dense_score": dense_score,
                    "attr_overlap": float(attribute_overlap)
                })
        
            scored_candidates.sort(key=lambda x: x["score"], reverse=True)
            return scored_candidates[:top_k]
        except Exception as e:
            logging.error(f"Error in search_and_rank: {e}")
            raise ProjectError(str(e), sys)

# singleton instance
ranker_instance = None

def get_ranker():
    global ranker_instance
    if ranker_instance is None:
        ranker_instance = Ranker()
    return ranker_instance
