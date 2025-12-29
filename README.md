# Cakeordog
Simple CLI classificator for "Muffin vs chihuahua" dataset https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification/data:
- train: train an SVM model and save it
- predict: predict a class for a single image using a saved model

## Development dependencies
```shell
sudo apt install shellcheck
sudo apt install shfmt
cargo install taplo-cli # lint-fmt for pyproject.toml
curl -LsSf https://astral.sh/uv/install.sh | sh # setup uv
uv sync
```

## Install
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
## Train
```
PYTHONPATH=src python -m cakeordog.cli train \
  --train-dir /path/to/data/train \
  --test-dir  /path/to/data/test \
  --model-out models/svc_muffin_chihuahua.joblib
```
## Predict
```
PYTHONPATH=src python -m cakeordog.cli predict \
  --model models/svc_muffin_chihuahua.joblib \
  --images img1.jpg img2.png
```
## Benchmark
OS: Kubuntu 24.04

RAM: 16GB

CPU: Intel i7-12700H

Results for test data from  
https://www.kaggle.com/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification/data

| Class       | Correct | Total | Accuracy |
|-------------|---------|-------|----------|
| chihuahua   | 532     | 640   | 0.8313   |
| muffin      | 449     | 544   | 0.8254   |
