#!/usr/bin/env bash
set -e

pip3 install -r requirements.txt

sudo systemctl start redis-server

# 3200 images downloaded and labeled
# python3 pipeline/pipeline_dataset.py

# model already trained uncomment to train new one
# python3 pipeline/pipeline_train.py

# db already indexed no need to re run alone  unless model also retrained
# python3 pipeline/pipeline_index.py

uvicorn app:app --reload