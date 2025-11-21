import json
import numpy as np
import pandas as pd
import umap
# import plotly.express as px # Không dùng vì output là JSON
import argparse
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to JSONL embeddings file')
    parser.add_argument('--titles', required=True, help='CSV file containing paper titles and DOI') 
    parser.add_argument('--output', default='output/umap_euclide_data.json', help='Output JSON file for FastAPI visualization') 
    parser.add_argument('--n-clusters', type=int, default=6, help='Number of clusters for topic grouping')
    parser.add_argument('--neighbors', type=int, default=15, help='UMAP n_neighbors parameter (lower = more local structure)')
    args = parser.parse_args()

    # -------------------- Load embeddings --------------------
    embeddings = []
    paper_ids = []

    print(f"Loading embeddings from {args.input} ...")
    with open(args.input, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            paper_ids.append(item['paper_id'])
            embeddings.append(item['embedding'])

    embeddings = np.array(embeddings)
    print(f"Loaded {len(embeddings)} papers with {embeddings.shape[1]}-dim embeddings")

    # -------------------- Load titles & DOI --------------------
    try:
        df_metadata = pd.read_csv(args.titles)
        
        if len(df_metadata) != len(embeddings):
            min_len = min(len(df_metadata), len(embeddings))
            df_metadata = df_metadata.head(min_len)
            embeddings = embeddings[:min_len]
            paper_ids = paper_ids[:min_len]

        titles = df_metadata['title'].tolist()
        dois = df_metadata['doi'].tolist()
        
    except Exception as e:
        print(f"Error loading metadata: {e}. Cannot proceed without DOI.")
        return 

    # -------------------- Run UMAP --------------------
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=min(args.neighbors, len(embeddings)-1),
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # -------------------- Clustering --------------------
    best_k = args.n_clusters
    
    if args.n_clusters is None: 
         print("Finding optimal number of clusters (k) using Silhouette Score...")
         pass

    print(f"Running KMeans clustering with {best_k} clusters on UMAP space (Euclidean)...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embedding_2d)
    
    silhouette_avg = silhouette_score(embedding_2d, cluster_labels)
    print(f"Final Silhouette Score: {silhouette_avg:.4f}")

    # -------------------- Prepare DataFrame and save to JSON --------------------

    df_vis = pd.DataFrame(embedding_2d, columns=['UMAP_1', 'UMAP_2'])
    df_vis['paper_id'] = paper_ids
    df_vis['title'] = titles
    df_vis['doi'] = dois
    
    df_vis['cluster'] = cluster_labels
    df_vis['cluster'] = 'Cluster ' + df_vis['cluster'].astype(str) 
    
    df_vis['url'] = 'https://www.biorxiv.org/content/' + df_vis['doi'].astype(str)

    print(f"Saving visualization data to {args.output}...")

    df_vis[['UMAP_1', 'UMAP_2', 'title', 'cluster', 'doi', 'url']].to_json(
        args.output,
        orient='records',
        lines=True,
        force_ascii=False 
    )
    
    print(f"Visualization data saved to {args.output}. Please run the FastAPI application.")

if __name__ == '__main__':
    main()