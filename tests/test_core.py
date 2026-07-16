import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import json

import os
import sys

# Append root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules after appending path
from config.attribute_schema import ATTR_TO_IDX, FLAT_ATTRIBUTE_LIST
from exception.exception import ProjectError
from logger.logger import logging

# ----------------- Tests for Config & Exceptions -----------------
def test_config():
    assert len(FLAT_ATTRIBUTE_LIST) == 75
    assert len(ATTR_TO_IDX) == 75
    assert ATTR_TO_IDX[FLAT_ATTRIBUTE_LIST[0]] == 0

def test_project_error():
    try:
        raise ProjectError("test error", sys)
    except ProjectError as e:
        assert "test error" in str(e)
        assert e.error_message == "test error"

# ----------------- Tests for Retriever Components -----------------
@patch('retriever.query_parser.get_embedder')
def test_query_parser(mock_get_embedder):
    from retriever.query_parser import QueryParser
    
    mock_embedder = MagicMock()
    # Return dummy embedding matrix for 75 attributes (each 10-dim for test)
    mock_embedder.embed_text.return_value = np.zeros((75, 10)).tolist()
    mock_get_embedder.return_value = mock_embedder
    
    parser = QueryParser()
    
    # Mock for a specific query
    mock_embedder.embed_text.return_value = np.zeros((10,)).tolist()
    
    vec = parser.parse_query("red shirt", top_k=5)
    assert vec.shape == (75,)
    assert np.all(vec >= 0)

@patch('retriever.search_db.get_embedder')
@patch('retriever.search_db.VectorStore')
def test_searcher(mock_vstore, mock_get_embedder):
    from retriever.search_db import Searcher
    
    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = [0.1, 0.2, 0.3]
    mock_get_embedder.return_value = mock_embedder
    
    mock_db = MagicMock()
    mock_db.query.return_value = {"ids": [["1"]], "distances": [[0.5]], "metadatas": [[{"image_path": "test.jpg"}]]}
    mock_vstore.return_value = mock_db
    
    searcher = Searcher()
    res = searcher.search("test")
    assert res["ids"] == [["1"]]

def test_cosine_overlap():
    from retriever.rank_images import cosine_overlap
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    assert np.isclose(cosine_overlap(a, b), 1.0)
    
    c = np.array([0, 1, 0])
    assert np.isclose(cosine_overlap(a, c), 0.0)

# ----------------- Tests for DB Vector Store -----------------
@patch('db.vector_store.chromadb.PersistentClient')
def test_vector_store(mock_client):
    from db.vector_store import VectorStore
    
    mock_chroma = MagicMock()
    mock_client.return_value = mock_chroma
    mock_collection = MagicMock()
    mock_chroma.get_or_create_collection.return_value = mock_collection
    
    vstore = VectorStore()
    
    vstore.upsert(["1"], [[0.1]], [{"path": "1.jpg"}])
    mock_collection.upsert.assert_called_once()
    
    vstore.query([[0.1]], 5)
    mock_collection.query.assert_called_once()

# ----------------- Tests for App endpoints -----------------
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

@patch('app.build_index')
def test_index_endpoint(mock_build):
    response = client.post("/index")
    assert response.status_code == 200
    assert response.json() == {"message": "Indexing started in the background."}

@patch('app.get_ranker')
@patch('app.redis_client')
def test_search_endpoint(mock_redis, mock_get_ranker):
    # Setup mock ranker
    mock_ranker = MagicMock()
    mock_parser = MagicMock()
    mock_parser.parse_query.return_value = np.zeros(75)
    mock_ranker.parser = mock_parser
    mock_ranker.search_and_rank.return_value = [{"id": "1", "score": 0.9}]
    mock_get_ranker.return_value = mock_ranker
    
    # Setup mock redis - simulate cache miss
    mock_redis.keys.return_value = []
    
    response = client.get("/search?q=test&top_k=5")
    assert response.status_code == 200
    assert response.json()["cached"] == False
    assert len(response.json()["results"]) == 1
