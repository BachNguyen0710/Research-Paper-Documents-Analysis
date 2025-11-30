# Research Paper Documents Analysis
This project establishes a complete, end-to-end pipeline for collecting, analyzing, and visualizing the topical structure of scholarly papers published on BioRxiv (the preprint repository for biological sciences). It leverages modern machine learning techniques (Embedding and UMAP) combined with the FastAPI web framework to deliver an interactive data exploration platform.

# How to run
## Require
Make sure you've installed all in requirement.txt
## 1. Embedding the data
options:
  * --data-path: đường dẫn data
  * --ouput: đường dẫn output
  * --batch_size: số lượng batch (should be 8 - 10)
  * --clean_stopword (loại bỏ stopword): True hoặc False
```bash
py src/embed_papers_hf.py --data-path data/biorxiv_sciedu.csv --output output/embeddings.jsonl --batch-size 8 --clean_stopword True
```
## 2. Visualization data
options:
  * --input: đường dẫn input
  * --titles: đường dẫn file data gốc (lấy titles)
  * --output: đường dẫn output
  * --n-clusters: tham số cluster
  * --neighbors: tham số neighbors (mặc định 15)
```bash
### Cosine Method
py src/umap_visualization-Cosine.py --input output/embeddings.jsonl --titles data/biorxiv_sciedu.csv --output umap_clusters-Consine.html --n-clusters 6
### Euclide Method
py src/umap_visualization-Euclide.py --input output/embeddings.jsonl --titles data/biorxiv_sciedu.csv --output umap_clusters-Euclide.html --n-clusters 6
### HDBSCAN
python src/umap_visualization_hdbscan.py --input output/embeddings.jsonl --titles data/biorxiv_sciedu.csv --output output/umap_hdbscan_clusters_fast.html
```
### 3. Extract data for FastApi
```bash
py src/umap_visualization-Euclide.py --input output/embeddings.jsonl --titles data/biorxiv_sciedu.csv --output output/umap_euclide_data.json --n-clusters 6
```
### 4. Run App
```bash
uvicorn app:app --reload
```
### Go to locall host http://127.0.0.1:8000 for visualization and interaction 
