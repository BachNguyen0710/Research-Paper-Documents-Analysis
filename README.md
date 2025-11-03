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
py src/umap_visualization.py --input output/embeddings.jsonl --titles data/biorxiv_sciedu.csv --output umap_clusters.html --n-clusters 6