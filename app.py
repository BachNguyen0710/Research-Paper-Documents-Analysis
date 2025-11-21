# app.py
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
from typing import List, Dict, Any

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

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main visualization page."""
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)