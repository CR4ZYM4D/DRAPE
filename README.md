<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=30&pause=1000&color=6366f1&center=true&vCenter=true&width=700&lines=DRAPE+-+AI+Fashion+Retrieval;Glance MLE Intern Assignment;Production-Grade+ML+Pipeline;End-to-End+MLOps+%7C+FastAPI+%7C+ChromaDB" alt="DRAPE | Glance MLE Intern Assignment" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-ML-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-FF6B35?style=for-the-badge&logo=chroma&logoColor=white)](https://trychroma.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Redis](https://img.shields.io/badge/Redis-Caching-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiment%20Tracking-000000?style=for-the-badge)](https://dagshub.com)

<br/>

> **A production-ready, end-to-end Machine Learning system** that performs multimodal fashion search and context retrieval using a fully automated training pipeline, experiment tracking, and a containerized REST API with semantic caching — built with engineering practices that match industry standards.

</div>

---

## About the Project

**DRAPE** is a state-of-the-art multimodal AI fashion retrieval system. Rather than relying on simple keyword matches, it utilizes a sophisticated two-stage retrieval architecture. It integrates **SigLIP** for zero-shot dense query embedding and a custom **EfficientNet-B0** classifier trained on a highly unbalanced multi-label attribute schema (75 features) to map complex, freeform fashion descriptions to the exact garments required.

This system is engineered for scale: decoupling the vector database for remote execution, caching semantic queries with Redis to bypass redundant inferences, and orchestrating entirely automated dataset, index, training, and inference pipelines. 

---

## Key Features

| Feature | Description |
|---|---|
| **Two-Stage Retrieval Pipeline** | Stage 1: Dense fast retrieval using SigLIP. Stage 2: Precision re-ranking using 75-dim soft probability attributes. |
| **Semantic Redis Caching** | Calculates `cosine_overlap` on queries. If overlap >= 95%, it instantly fetches sub-millisecond cached responses, saving heavy DB reads. |
| **Automated MLOps Pipelines** | Granular execution via `pipeline_dataset.py`, `pipeline_train.py`, `pipeline_index.py`, and `pipeline_infer.py`. |
| **Model Training & Balancing** | Custom PyTorch EfficientNet architecture, trained using BCEWithLogitsLoss and `pos_weight` to perfectly handle multi-label class imbalances. |
| **Experiment Tracking** | MLflow + DagsHub integration for F1/Precision/Recall metrics, parameter logging, and Pickle serialization versioning. |
| **Decoupled Vector DB** | ChromaDB architecture built for scalability, utilizing `HttpClient` for seamless routing to remote cloud instances. |
| **FastAPI Inference + UI** | Production-grade `/search` endpoint coupled with a premium, responsive glassmorphism UI offering dynamic slider constraints. |
| **Dockerized** | Fully containerized app ready for AWS (ECS/EKS/EC2) or any cloud deployment. |
| **Robust Logging & Exceptions** | Centralized structured `logger` and customized `ProjectError` for strict traceback control. |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MLOps Pipelines                        │
│                                                             │
│   Raw Image Data ──► Dataset Prep ──►  Model Training       │
│                                              │              │
│                                   PyTorch EfficientNet-B0   │
│                                              │              │
│                      MLflow / DagsHub Model Registry        │
│                                              │              │
│   Local/Cloud Images ──► Vector Indexing ◄───┘              │
│                                │                            │
│                      Remote ChromaDB Server                 │
└────────────────────────────────┼────────────────────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │       FastAPI API        │
                    │                          │
User Query ─────────►  1. Redis Cache Check    │
                    │  2. SigLIP Dense Search  │
                    │  3. Soft-Attr Re-ranking │
                    │                          │
                    └────────────┬─────────────┘
                                 │
                      JSON Response / Premium UI
```

---

## Project Structure

```
DRAPE/
│
├── app.py                    # FastAPI application entrypoint and semantic cache logic
├── index.html                # Premium UI interface
├── Dockerfile                # Container definition
├── requirements.txt          # All dependencies
├── setup.py                  # Package setup
├── infer.sh                  # All-in-one local bash script to boot Redis, prep data, and launch API
│
├── pipeline/                 # MLOps Orchestration Pipelines
│   ├── pipeline_dataset.py   # Data preparation and formatting
│   ├── pipeline_train.py     # Training and MLflow tracking
│   ├── pipeline_index.py     # Indexing to remote/local DB
│   └── pipeline_infer.py     # Downloads latest model and boots server
│
├── data/                     # Raw processing and label management
│   ├── label_data.py
│   ├── generate_queries.py
│   └── download_dataset.py
│
├── models/                   
│   └── attribute_classifier.py # PyTorch EfficientNet implementation
│
├── training/
│   └── train_classifier.py   # Training loop with class-weighting and evaluation
│
├── indexer/                  # Vector ingestion logic
│   ├── embed_image.py        
│   ├── classify_image.py
│   └── build_index.py
│
├── retriever/                # Real-time search and parsing
│   ├── query_parser.py       # Converts freeform text to vectors via SigLIP
│   ├── search_db.py          # Stage 1 Chroma DB retrieval
│   └── rank_images.py        # Stage 2 Attribute Re-ranking
│
├── db/                       
│   └── vector_store.py       # ChromaDB persistent/HTTP client abstraction
│
├── config/                   
│   └── attribute_schema.py   # Global 75-dim attribute map
│
├── logger/                   # Custom structured logger
├── exception/                # ProjectError handler
├── tests/                    # Pytest coverage files
└── .github/workflows/        # Automated CI/CD
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Docker (optional, recommended for production)
- Redis Server (for semantic caching)
- DagsHub account (for MLflow tracking)

### 1. Clone & Install

```bash
git clone https://github.com/CR4ZYM4D/DRAPE.git
cd DRAPE
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Vector Database
CHROMA_HOST=                    # Leave blank for local persistent storage
CHROMA_PORT=8000
CHROMA_PATH=chroma_db

# Caching
REDIS_URL=redis://localhost:6379

# MLflow Tracking
MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/DRAPE.mlflow
MLFLOW_TRACKING_USERNAME=<dagshub-username>
MLFLOW_TRACKING_PASSWORD=<dagshub-token>

# Configs
DATASTORE_PATH=data/raw/images
NUM_EPOCHS=10
EXHAUSTIVE_THRESHOLD=500
```

### 3. Run the Application Locally (Recommended)

The easiest way to bootstrap the entire environment locally is using the included `infer.sh` bash script. This script automatically:
1. Installs all Python dependencies.
2. Boots up the Redis server for semantic caching.
3. Prepares the dataset.
4. Builds the Vector DB Index.
5. Boots the FastAPI Uvicorn server.

```bash
chmod +x infer.sh
./infer.sh
```

*(Alternatively, you can manually run `python pipeline/pipeline_infer.py` to only fetch MLflow weights and start the API without dataset rebuilding).*

### 4. Or Deploy via Docker (AWS ECS/EC2)

```bash
docker build -t drape-search-api .
docker run -p 8000:8000 --env-file .env drape-search-api
```

---

## API Reference

Once running, the interactive Swagger docs are available at **`http://localhost:8000/docs`**
The Premium Web UI is available at **`http://localhost:8000/`**

### `GET /search`
Fetch visually and semantically ranked fashion items based on a freeform query string.

```bash
curl -X GET "http://localhost:8000/search?q=a%20person%20standing%20in%20rain%20wearing%20a%20yellow%20raincoat&top_k=5"
```

**Response:**
```json
{
  "query": "a person standing in rain wearing a yellow raincoat",
  "cached": true,
  "results": [
    {
      "image_path": "data/raw/images/123.jpg",
      "score": 0.942
    },
    ...
  ]
}
```

### `POST /index`
Triggers the asynchronous vector indexing pipeline to index new garments on the fly.

```bash
curl -X POST "http://localhost:8000/index"
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10 |
| **ML Framework** | PyTorch, torchvision, timm |
| **Vision/Embeddings** | SigLIP (HuggingFace Transformers) |
| **API & Frontend** | FastAPI, HTML5/CSS3 (Glassmorphism) |
| **Database** | ChromaDB (Vector) |
| **Caching** | Redis |
| **Experiment Tracking** | MLflow + DagsHub |
| **Containerization** | Docker |
| **Testing & CI/CD** | Pytest, GitHub Actions |

---

## ML Pipeline Deep Dive

### Data & Indexing
Unstructured fashion images are processed into two modalities: Dense vectors (via `google/siglip-base-patch16-224`) and 75-dim soft probability vectors (via custom EfficientNet-B0). These are packaged as metadata and upserted into ChromaDB via HTTP or locally via persistent storage.

### Model Training (`pipeline_train.py`)
Because fashion tags are highly unbalanced, the PyTorch model calculates dataset statistics dynamically and applies a `pos_weight` matrix to a `BCEWithLogitsLoss` criterion. Metrics like Precision, Recall, and F1-Score are logged to MLFlow. If the newly trained model beats the current F1 High Score, it automatically transitions to "Production" in the registry.

### Semantic Search & Caching (`pipeline_infer.py` & `app.py`)
Queries arrive at the FastAPI server and are first vectorized by SigLIP. The server checks the Redis Cache for historically overlapping intent (`cosine_overlap > 0.90`). If there's a hit with sufficient results, it returns instantly. If not, it falls back to ChromaDB for Stage-1 retrieval, runs a Stage-2 rigorous attribute dot-product sort, caches the new result, and serves it back to the user.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

If this project helped you, consider leaving a star!

</div>
