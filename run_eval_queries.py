import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from retriever.rank_images import get_ranker
from logger.logger import logging
from exception.exception import ProjectError

def run_eval():
    try:
        if not os.path.exists("data/sample_queries.json"):
            logging.error("No sample queries found. Run data/generate_queries.py first.")
            return
            
        with open("data/sample_queries.json", "r") as f:
            queries = json.load(f)
            
        ranker = get_ranker()
        
        logging.info("Running evaluation queries...")
        for i, q in enumerate(queries[:5]): # Just run first 5 for quick check
            query_text = q["query"]
            logging.info(f"\nQuery {i+1}: '{query_text}'")
            results = ranker.search_and_rank(query_text, top_k=3)
            
            if not results:
                logging.info("  No results found.")
            for r in results:
                logging.info(f"  [{r['score']:.3f}] {r['image_path']}")
                logging.info(f"    Dense: {r['dense_score']:.3f}, Attr: {r['attr_overlap']:.3f}")
    except Exception as e:
        logging.error(f"Error running evaluation queries: {e}")
        raise ProjectError(str(e), sys)
                
if __name__ == "__main__":
    run_eval()
