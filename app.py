from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import redis
import json
import uuid
import os
import sys
import numpy as np
from dotenv import load_dotenv

from indexer.build_index import build_index
from retriever.rank_images import get_ranker, cosine_overlap
from logger.logger import logging
from exception.exception import ProjectError

load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

try:
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    # Test connection
    redis_client.ping()
    logging.info(f"Connected to Redis at {REDIS_URL}")
except Exception as e:
    logging.warning(f"Failed to connect to Redis. Caching will be disabled. Error: {e}")
    redis_client = None

app = FastAPI(title="DRAPE Multimodal Retrieval API")

# Serve the image directory statically so the frontend can fetch them
if os.path.exists(os.getenv("DATASTORE_PATH", "data/raw/images")):
    app.mount("/data/raw/images", StaticFiles(directory=os.getenv("DATASTORE_PATH", "data/raw/images")), name="images")

@app.get("/")
async def serve_frontend():
    """Serves the premium UI."""
    frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"message": "index.html not found!"}

@app.post("/index")
async def index_data(background_tasks: BackgroundTasks):
    """Triggers the indexing pipeline in the background."""
    try:
        background_tasks.add_task(build_index)
        logging.info("Indexing started in the background.")
        return {"message": "Indexing started in the background."}
    except Exception as e:
        logging.error(f"Error starting index: {e}")
        raise ProjectError(str(e), sys)

@app.get("/search")
async def search(q: str, top_k: int = Query(5, ge=1, le=50)):
    """Searches the indexed data using the two-stage retrieval pipeline with semantic caching."""
    try:
        logging.info(f"Received search query: '{q}' for top_k={top_k}")
        ranker = get_ranker()
        
        # 1. Parse query to attribute vector
        query_attr_vec = ranker.parser.parse_query(q)
        
        # 2. Check Semantic Cache
        if redis_client:
            try:
                keys = redis_client.keys("query_cache:*")
                for key in keys:
                    cached_data_str = redis_client.get(key)
                    if not cached_data_str: continue
                    
                    cached_data = json.loads(cached_data_str)
                    if cached_data["top_k"] < top_k:
                        continue # Cache doesn't have enough results
                        
                    cached_attr_vec = np.array(cached_data["query_attr_vec"])
                    overlap = cosine_overlap(query_attr_vec, cached_attr_vec)
                    
                    if overlap >= 0.90:
                        logging.info(f"Semantic Cache Hit! Overlap: {overlap:.4f} with cached query '{cached_data['query']}'")
                        return {"query": q, "results": cached_data["results"][:top_k], "cached": True}
            except Exception as cache_err:
                logging.warning(f"Error reading from Redis cache: {cache_err}")
                
        # 3. Cache Miss - Run Retrieval
        logging.info("Cache miss. Running full retrieval...")
        results = ranker.search_and_rank(q, top_k=top_k)
        
        # 4. Save to Cache
        if redis_client:
            try:
                cache_key = f"query_cache:{uuid.uuid4()}"
                cache_payload = {
                    "query": q,
                    "query_attr_vec": query_attr_vec.tolist(),
                    "top_k": top_k,
                    "results": results
                }
                # Store with 1 hour expiration
                redis_client.setex(cache_key, 3600, json.dumps(cache_payload))
            except Exception as cache_err:
                logging.warning(f"Error saving to Redis cache: {cache_err}")
                
        return {"query": q, "results": results, "cached": False}
        
    except Exception as e:
        logging.error(f"Error in search endpoint: {e}")
        raise ProjectError(str(e), sys)

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except Exception as e:
        logging.error(f"Error starting uvicorn: {e}")
        raise ProjectError(str(e), sys)
