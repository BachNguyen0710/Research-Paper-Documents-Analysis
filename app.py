# app.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import defaultdict
import numpy as np

app = FastAPI()

templates = Jinja2Templates(directory="templates")


DATA_FILE = "output/umap_euclide_data.json"

@app.get("/api/data", response_model=Dict[str, List[Dict[str, Any]]])
async def get_visualization_data():
    """Load and return the UMAP visualization data from the specified JSON file."""
    if not os.path.exists(DATA_FILE):
        return {"error": "Visualization data file not found. Did you run the Python script first?"}, 404
    
    try:
        df = pd.read_json(DATA_FILE, orient='records', lines=True)
        data_for_frontend = df.to_dict('records')
        
        return {"data": data_for_frontend}
        
    except Exception as e:
        return {"error": f"Error processing data file: {str(e)}"}, 500

@app.get("/api/cluster-keywords")
async def get_cluster_keywords():
    """Generate TF-IDF keywords for each cluster."""
    if not os.path.exists(DATA_FILE):
        return {"error": "Visualization data file not found."}, 404
    
    try:
        df = pd.read_json(DATA_FILE, orient='records', lines=True)
        
        # Kiểm tra xem có cột abstract không
        if 'abstract' not in df.columns:
            return {"error": "Abstract data not found. Please regenerate data with abstracts."}, 400
        
        cluster_texts = defaultdict(list)
        for _, row in df.iterrows():
            if pd.notna(row['abstract']) and row['abstract'].strip():
                cluster_texts[row['cluster']].append(row['abstract'])
        
        if not cluster_texts:
            return {"error": "No valid abstracts found."}, 400
        
        cluster_docs = {}
        for cluster, abstracts in cluster_texts.items():
            cluster_docs[cluster] = ' '.join(abstracts)
        
        # Tiền xử lý text với custom stopwords
        def preprocess_text(text):
            text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
            
            basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
                             'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 
                             'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 
                             'those', 'we', 'us', 'our', 'ours', 'you', 'your', 'yours', 'he', 'him', 'his', 
                             'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs', 'not', 'no', 
                             'yes', 'one', 'two', 'three', 'first', 'second', 'third', 'also', 'more', 'most', 
                             'other', 'some', 'such', 'only', 'all', 'any', 'each', 'both', 'between', 'through'}
            
            scientific_stopwords = {'study', 'studies', 'research', 'method', 'methods', 'analysis', 'data', 
                                   'result', 'results', 'conclusion', 'conclusions', 'background', 'objective', 
                                   'objectives', 'purpose', 'approach', 'paper', 'article', 'work', 'present', 
                                   'show', 'showed', 'shown', 'found', 'find', 'finding', 'findings', 'observed', 
                                   'observe', 'suggest', 'suggests', 'suggested', 'indicate', 'indicates', 
                                   'indicated', 'demonstrate', 'demonstrates', 'demonstrated', 'report', 'reported', 
                                   'describe', 'described', 'investigate', 'investigated', 'examine', 'examined', 
                                   'evaluate', 'evaluated', 'assess', 'assessed', 'analyze', 'analyzed', 'measure', 
                                   'measured', 'compare', 'compared', 'comparison', 'based', 'using', 'used', 
                                   'significantly', 'significant', 'different', 'difference', 'differences', 
                                   'important', 'potential', 'possible', 'likely', 'however', 'therefore', 
                                   'moreover', 'furthermore', 'additionally', 'finally', 'conclusion', 'abstract', 
                                   'introduction', 'discussion', 'experimental', 'experiment', 'experiments',
                                   'including', 'respectively', 'particularly', 'especially', 'mainly', 'generally',
                                   'specifically', 'overall', 'total', 'number', 'level', 'levels', 'effect', 
                                   'effects', 'factor', 'factors', 'model', 'models', 'system', 'systems',
                                   'process', 'processes', 'well', 'high', 'low', 'large', 'small', 'new', 'old'}
            
            medical_stopwords = {'patient', 'patients', 'clinical', 'trial', 'trials', 'treatment', 'treatments', 
                               'therapy', 'therapies', 'disease', 'diseases', 'condition', 'conditions', 'cell', 
                               'cells', 'gene', 'genes', 'protein', 'proteins', 'sample', 'samples', 'group', 
                               'groups', 'control', 'controls', 'case', 'cases', 'subject', 'subjects', 
                               'individual', 'individuals', 'population', 'populations', 'age', 'gender', 
                               'male', 'female', 'year', 'years', 'time', 'times', 'week', 'weeks', 'day', 
                               'days', 'month', 'months', 'hour', 'hours', 'percent', 'percentage'}
            
            all_stopwords = basic_stopwords | scientific_stopwords | medical_stopwords
            
            words = [word for word in text.split() if word not in all_stopwords and len(word) > 2]
            return ' '.join(words)
        
        processed_docs = {cluster: preprocess_text(text) for cluster, text in cluster_docs.items()}
        
        # Tính TF-IDF với tham số được tối ưu
        vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 3),  
            min_df=1,  
            max_df=0.7, 
            stop_words=None,  
            lowercase=False,  
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        cluster_names = list(processed_docs.keys())
        documents = [processed_docs[cluster] for cluster in cluster_names]
        
        if not documents or not any(doc.strip() for doc in documents):
            return {"error": "No valid processed documents found."}, 400
        
        tfidf_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        def generate_cluster_label(keywords):
            if not keywords:
                return "Unknown Topic"
            
            top_keywords = [kw['keyword'] for kw in keywords[:3]]

            topic_patterns = {
                'Genetics & Genomics': ['genetic', 'genome', 'dna', 'mutation', 'sequencing', 'genomic', 'variant', 'chromosome', 'inheritance', 'allele'],
                'Cancer Research': ['cancer', 'tumor', 'malignant', 'oncology', 'chemotherapy', 'metastasis', 'carcinoma', 'neoplasm', 'radiation', 'biopsy'],
                'Neuroscience': ['brain', 'neural', 'neuron', 'cognitive', 'neurological', 'memory', 'learning', 'synaptic', 'cortex', 'alzheimer'],
                'Immunology': ['immune', 'antibody', 'vaccination', 'antigen', 'lymphocyte', 'inflammatory', 'autoimmune', 'cytokine', 'immunization'],
                'Cardiology': ['heart', 'cardiac', 'cardiovascular', 'blood pressure', 'coronary', 'myocardial', 'vessel', 'circulation', 'stroke'],
                'Infectious Diseases': ['infection', 'bacterial', 'viral', 'pathogen', 'antimicrobial', 'epidemic', 'vaccine', 'antibiotic', 'contamination'],
                'Cell Biology': ['cellular', 'mitochondrial', 'membrane', 'organelle', 'cytoplasm', 'nucleus', 'division', 'apoptosis', 'metabolism'],
                'Molecular Biology': ['molecular', 'enzyme', 'binding', 'interaction', 'pathway', 'signaling', 'regulation', 'expression', 'synthesis'],
                'Pharmacology': ['drug', 'therapeutic', 'dosage', 'pharmacokinetic', 'toxicity', 'medication', 'pharmaceutical', 'clinical trial'],
                'Epidemiology': ['population', 'prevalence', 'incidence', 'risk factor', 'cohort', 'surveillance', 'outbreak', 'mortality', 'morbidity'],
                'Data Science & AI': ['machine learning', 'algorithm', 'prediction', 'classification', 'deep learning', 'artificial intelligence', 'model', 'dataset'],
                'Medical Imaging': ['imaging', 'scan', 'mri', 'tomography', 'ultrasound', 'radiological', 'diagnostic', 'visualization', 'reconstruction'],
                'Public Health': ['health policy', 'prevention', 'community health', 'healthcare', 'intervention', 'screening', 'awareness', 'promotion']
            }
            
            keyword_text = ' '.join(top_keywords).lower()
            best_match = None
            max_matches = 0
            
            for topic, patterns in topic_patterns.items():
                matches = sum(1 for pattern in patterns if pattern in keyword_text)
                if matches > max_matches:
                    max_matches = matches
                    best_match = topic
            

            if best_match and max_matches >= 1:
                return best_match
            
            if len(top_keywords) >= 2:
                return f"{top_keywords[0].title()} & {top_keywords[1].title()}"
            else:
                return top_keywords[0].title() if top_keywords else "General Research"

        cluster_keywords = {}
        cluster_labels = {}
        for i, cluster in enumerate(cluster_names):
            tfidf_scores = tfidf_matrix[i].toarray()[0]
            
            top_indices = np.argsort(tfidf_scores)[-20:][::-1]
            keywords = [
                {
                    "keyword": feature_names[idx],
                    "score": float(tfidf_scores[idx])
                }
                for idx in top_indices if tfidf_scores[idx] > 0.01
            ]
            
            cluster_keywords[cluster] = keywords[:15]
            
            cluster_labels[cluster] = generate_cluster_label(keywords[:15])
        
        return {
            "keywords": cluster_keywords,
            "labels": cluster_labels
        }
        
    except Exception as e:
        return {"error": f"Error processing TF-IDF: {str(e)}"}, 500

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main visualization page."""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)