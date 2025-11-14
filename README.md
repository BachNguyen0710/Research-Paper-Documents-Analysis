# How to run
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