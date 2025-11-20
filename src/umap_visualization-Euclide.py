import json
import numpy as np
import pandas as pd
import umap
import plotly.express as px
import argparse
from tqdm.auto import tqdm
from sklearn.cluster import KMeans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to JSONL embeddings file')
    parser.add_argument('--titles', required=False, help='Optional CSV file containing paper titles for hover info')
    #parser.add_argument('--output', default='umap_clusters.html', help='Output HTML file for visualization')
    parser.add_argument('--output', default = 'output/map_euclide_data.json', help='Output JSON file for FastAPI visualization')
    parser.add_argument('--n-clusters', type=int, default=5, help='Number of clusters for topic grouping')
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

    # -------------------- Load titles --------------------
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
        print(f"Error loading metadata: {e}. Using default titles/DOIs.")
        titles = [f"Paper {i}" for i in range(len(embeddings))]
        dois = [f"10.1101/Unknown_{i}" for i in range(len(embeddings))]

    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=min(args.neighbors, len(embeddings)-1),
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # -------------------- Clustering --------------------
    print(f"Running KMeans clustering with {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    df_vis = pd.DataFrame(embedding_2d, columns=['UMAP_1', 'UMAP_2'])
    df_vis['paper_id'] = paper_ids
    df_vis['title'] = titles
    df_vis['cluster'] = cluster_labels
    df_vis['cluster'] = 'Cluster ' + df_vis['cluster'].astype(str) 
    df_vis['doi'] = dois
    df_vis['url'] = 'https://www.biorxiv.org/content/' + df_vis['doi'].astype(str)

    print(f"Saving visualization data to {args.output}...")
    df_vis[['UMAP_1', 'UMAP_2', 'title', 'cluster', 'doi', 'url']].to_json(
        args.output,
        orient='records',
        lines=True,
        force_ascii=False 
    )
    
    print(f"Visualization data saved to {args.output}. Please run the FastAPI application.")
if __name__ == "__main__":
    main()