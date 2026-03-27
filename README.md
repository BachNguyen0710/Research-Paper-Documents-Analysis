# Research Paper Documents Analysis System

An end-to-end pipeline for collecting, analyzing, and visualizing the topical structure of scholarly papers from **BioRxiv**, leveraging **Machine Learning (Specter Embedding & UMAP)** and a **FastAPI** web framework.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![Machine Learning](https://img.shields.io/badge/ML-Specter%20%26%20UMAP-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)

---

## 1. Introduction

This project establishes a complete workflow to analyze biological science preprints from BioRxiv. The system allows users to explore relationships between thousands of research papers through high-dimensional vector spaces and automated clustering.

### 1.1. Key Features
- 🤖 **Domain-Specific Embedding**: Utilizes the `allenai/specter` model, specifically optimized for representing scientific documents.
- 🗺️ **Interactive Visualization**: Employs the **UMAP** algorithm to reduce data dimensionality and create interactive 2D/3D knowledge maps.
- 🏷️ **Automated Topic Labeling**: 
    - Extracts keywords using **TF-IDF** for each identified cluster.
    - Automatically identifies research fields such as: *Genetics, Cancer Research, Neuroscience, Immunology...*.
- 📚 **API Documentation**: 
    - Automated Swagger UI available at `/docs`.
- 🚀 **Complete Data Pipeline**: 
    - Text preprocessing (including removal of specialized biomedical stopwords).
    - Generation of embedding vectors and data clustering.

### 1.2. Project Purpose
This tool was developed to help researchers quickly grasp trends and topical structures within the massive repository of BioRxiv scholarly data.

---

## 2. Project Structure

```text
Research-Paper-Documents-Analysis/
├── app/                        # Web Application
│   ├── app.py                  # FastAPI Backend (Logic & TF-IDF)
│   ├── templates/              # HTML Templates
│   │   └── index.html          # Main Visualization Dashboard
│   └── static/                 # CSS, JS, and Static Assets
├── data/                       # Project Data
│   ├── biorxiv_sciedu.csv      # Original paper dataset
│   └── Crawling_data.ipynb     # Data collection notebook
├── src/                        # Source Code (Core Processing)
│   ├── embed_papers_hf.py      # Script to generate embeddings via Specter
│   ├── umap_visualization-Euclide.py  # Euclidean distance visualization
│   ├── umap_visualization-Cosine.py   # Cosine distance visualization
│   └── umap_visualization_hdbscan.py  # HDBSCAN density clustering
├── output/                     # Output Results
│   ├── embeddings.jsonl        # Stored embedding vectors
│   └── umap_euclide_data.json  # Final data for Web App
├── Dockerfile                  # Docker packaging configuration
├── requirements.txt            # Python library dependencies
├── .gitignore                  # Excluded files/folders
└── README.md                   # Project documentation# Research Paper Documents Analysis
This project establishes a complete, end-to-end pipeline for collecting, analyzing, and visualizing the topical structure of scholarly papers published on BioRxiv (the preprint repository for biological sciences). It leverages modern machine learning techniques (Embedding and UMAP) combined with the FastAPI web framework to deliver an interactive data exploration platform.
```
## 2. Embedding the data
options:
  * --data-path: path to data
  * --ouput: path output
  * --batch_size: numbers of batch (should be 8 - 10)
  * --clean_stopword (remove stopword): True hoặc False
```bash
py src/embed_papers_hf.py --data-path data/biorxiv_sciedu.csv --output output/embeddings.jsonl --batch-size 8 --clean_stopword True
```
## 3. Visualization data
options:
```bash
### Cosine Method
py src/umap_visualization-Cosine.py --input output/embeddings.jsonl --titles data/biorxiv_sciedu.csv --output umap_clusters-Consine.html --n-clusters 6
### Euclide Method
py src/umap_visualization-Euclide.py --input output/embeddings.jsonl --titles data/biorxiv_sciedu.csv --output umap_clusters-Euclide.html --n-clusters 6
### HDBSCAN
python src/umap_visualization_hdbscan.py --input output/embeddings.jsonl --titles data/biorxiv_sciedu.csv --output output/umap_hdbscan_clusters_fast.html
```
## 4. Extract data for FastApi
```bash
py src/umap_visualization-Euclide.py --input output/embeddings.jsonl --titles data/biorxiv_sciedu.csv --output output/umap_euclide_data.json --n-clusters 6
```
## 5. Run App (local)
```bash
uvicorn app.app:app --reload
```
### Go to locall host http://127.0.0.1:8000 for visualization and interaction

### 6. Run Docker 
1.  **Build image**:
```bash 
docker build -t taxi-app:latest .
```

2.  **Chạy container**:
```bash
docker run -d -p 8000:8000 --name taxi-container taxi-app:latest
```

*   \-d → chạy ở background
    
*   \-p 8000:8000 → map port host → container
    
*   \--name taxi-container → đặt tên container
    

3.  **Kiểm tra logs** (nếu muốn xem output):

```bash
docker logs -f taxi-container
```
4. Mở trình duyệt và truy cập: `http://localhost:8000`

5. Dừng container khi không sử dụng:
```bash
docker stop taxi-container
```

6. **Ở những lần chạy sau, chỉ cần truy cập trên `http://localhost:8000`**.
* Nếu container đã bị stop, chạy lại container:
  ```bash
  docker start taxi-container
  ```
* Kiểm tra container đã chạy chưa:
  ```bash
  docker ps
  ```

