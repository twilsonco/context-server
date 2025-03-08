from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import logging

logger = logging.getLogger(__name__)

from .config import config, CONFIG_PATH
from .vector_store import VectorStore
from .file_watcher import MDWatcher

app = FastAPI(title="Markdown Vector Search", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global state
logger.info("Initializing vector store...")
vector_store = VectorStore()
logger.info("Vector store initialized")

logger.info(f"Setting up file watcher for directory: {config['docs_dir']}")
watcher = MDWatcher(os.path.abspath(config["docs_dir"]), vector_store)
last_full_index_time = None
last_change_index_time = None

@app.on_event("startup")
def on_startup():
    """Initialize the application on startup."""
    global last_full_index_time, last_change_index_time
    logger.info("Application startup beginning...")
    
    logger.info("Ensuring docs directory exists...")
    os.makedirs(config["docs_dir"], exist_ok=True)
    
    logger.info("Starting initial indexing of existing files...")
    watcher.index_all()
    
    last_full_index_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_change_index_time = last_full_index_time
    
    logger.info("Starting file watcher...")
    watcher.start()
    logger.info("Application startup complete")

@app.on_event("shutdown")
def on_shutdown():
    """Clean up on application shutdown."""
    logger.info("Application shutting down...")
    watcher.stop()
    logger.info("Shutdown complete")

@app.get("/api/settings", response_class=JSONResponse)
def get_settings():
    """Get current settings and indexing status."""
    indexing_status = vector_store.get_indexing_status()
    return {
        "settings": {
            "docs_dir": config["docs_dir"],
            "timezone": config["timezone"],
            "include_titles": config["include_titles"],
            "retrieval_mode": config["retrieval_mode"],
            "recency_weight": config["recency_weight"],
            "n_candidates": config["n_candidates"],
            "n_results": config["n_results"]
        },
        "status": {
            "last_full_index_time": last_full_index_time,
            "last_change_index_time": last_change_index_time,
            "indexed_segments": {
                "day": len(vector_store.id_to_doc['day']),
                "memory": len(vector_store.id_to_doc['memory']),
                "section": len(vector_store.id_to_doc['section']),
                "line": len(vector_store.id_to_doc['line'])
            },
            "indexing": indexing_status
        }
    }

@app.post("/api/settings", response_class=JSONResponse)
def update_settings(new_settings: dict):
    """Update application settings."""
    allowed = {
        "timezone", "include_titles", "retrieval_mode",
        "recency_weight", "n_candidates", "n_results"
    }
    
    for key, value in new_settings.items():
        if key in allowed:
            if key == "include_titles":
                config[key] = bool(value)
            elif key in ("n_candidates", "n_results"):
                config[key] = int(value)
            elif key == "recency_weight":
                config[key] = float(value)
            else:
                config[key] = value
            if key == "timezone":
                try:
                    from zoneinfo import ZoneInfo
                    TZ = ZoneInfo(value)
                    config["timezone"] = value
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid timezone: {e}")

    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Could not save config: {e}")

    return {
        "settings": config,
        "message": "Settings updated. Some changes may require an index refresh."
    }

@app.get("/api/query", response_class=JSONResponse)
def query(q: str, mode: str = None, recency_weight: float = None, n_results: int = None):
    """Search the vector store."""
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' cannot be empty.")
    
    results = vector_store.search(
        q.strip(),
        mode=mode,
        recency_weight=recency_weight,
        n_results=n_results
    )
    
    return {
        "query": q.strip(),
        "mode": mode or config["retrieval_mode"],
        "results": results
    }

@app.post("/api/refresh", response_class=JSONResponse)
def refresh_index():
    """Re-index all files."""
    global last_full_index_time, last_change_index_time
    vector_store.reset()
    watcher.index_all()
    last_full_index_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_change_index_time = last_full_index_time
    return {
        "message": "Index refreshed successfully.",
        "last_full_index_time": last_full_index_time
    }

@app.post("/api/reset", response_class=JSONResponse)
def reset_index():
    """Clear the index."""
    global last_full_index_time, last_change_index_time
    vector_store.reset()
    last_full_index_time = None
    last_change_index_time = None
    return {
        "message": "Index cleared. Use refresh to re-index files."
    }

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the web UI."""
    indexing_status = vector_store.get_indexing_status()
    return f"""
    <html>
      <head>
        <title>Markdown Search Settings</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 2em; }}
          h1 {{ color: #333; }}
          .section {{ margin-bottom: 2em; }}
          label {{ display: inline-block; width: 130px; font-weight: bold; }}
          .field {{ margin: 0.3em 0; }}
          #results {{ margin-top: 1em; padding-top: 1em; border-top: 1px solid #ccc; }}
          .result-item {{ margin-bottom: 1.5em; }}
          .result-item p {{ margin: 0.2em 0; }}
          .result-meta {{ font-size: 0.9em; color: #555; }}
          .progress {{ width: 100%; background-color: #f0f0f0; }}
          .progress-bar {{ height: 20px; background-color: #4CAF50; text-align: center; line-height: 20px; color: white; }}
          .warning {{ color: #856404; background-color: #fff3cd; border: 1px solid #ffeeba; padding: 0.75rem 1.25rem; margin-bottom: 1rem; border-radius: 0.25rem; }}
        </style>
      </head>
      <body>
        <h1>Markdown Search Settings</h1>
        <div class="section">
          <h2>Status</h2>
          <p>Real-time indexing: <strong>Active</strong></p>
          <p>Last full index refresh: <strong>{last_full_index_time or "N/A"}</strong></p>
          <p>Last file indexed change: <strong>{last_change_index_time or "N/A"}</strong></p>
          <p>Indexed segments: Day={len(vector_store.id_to_doc['day'])}, Memory={len(vector_store.id_to_doc['memory'])}, Section={len(vector_store.id_to_doc['section'])}, Line={len(vector_store.id_to_doc['line'])}</p>
          
          {"" if not indexing_status['is_indexing'] else f'''
          <div class="progress">
            <div class="progress-bar" style="width: {indexing_status['progress']}%">
              {int(indexing_status['progress'])}%
            </div>
          </div>
          <p>Currently indexing: {indexing_status['current_file'] or 'N/A'}</p>
          <p>Files processed: {indexing_status['processed_files']}/{indexing_status['total_files']}</p>
          '''}
          
          {f'<p class="warning">Last indexing error: {indexing_status["error"]}</p>' if indexing_status.get('error') else ''}
        </div>
        
        <div class="section">
          <h2>Configuration</h2>
          <div class="field">
            <label>Timezone:</label>
            <input type="text" id="tz" value="{config["timezone"]}" size="20"/>
          </div>
          <div class="field">
            <label>Include Titles:</label>
            <select id="include_titles">
              <option value="true" {"selected" if config["include_titles"] else ""}>True</option>
              <option value="false" {"selected" if not config["include_titles"] else ""}>False</option>
            </select>
          </div>
          <div class="field">
            <label>Default Mode:</label>
            <select id="mode">
              <option value="day" {"selected" if config["retrieval_mode"]=="day" else ""}>Whole Day</option>
              <option value="memory" {"selected" if config["retrieval_mode"]=="memory" else ""}>Memories (#)</option>
              <option value="section" {"selected" if config["retrieval_mode"]=="section" else ""}>Sections (##)</option>
              <option value="line" {"selected" if config["retrieval_mode"]=="line" else ""}>Lines (&gt;)</option>
            </select>
          </div>
          <div class="field">
            <label>Recency Weight:</label>
            <input type="number" id="recency" step="0.1" value="{config["recency_weight"]}" />
          </div>
          <div class="field">
            <label>Candidates (N):</label>
            <input type="number" id="n_candidates" value="{config["n_candidates"]}" />
          </div>
          <div class="field">
            <label>Results (K):</label>
            <input type="number" id="n_results" value="{config["n_results"]}" />
          </div>
          <button onclick="saveSettings()">Save Settings</button>
        </div>
        <div class="section">
          <h2>Maintenance</h2>
          <button onclick="refreshIndex()">Refresh Index</button>
          <button onclick="confirmReset()">Reset Index</button>
        </div>
        <div class="section">
          <h2>Test Query</h2>
          <input type="text" id="query_input" size="40" placeholder="Enter search query..."/>
          <select id="query_mode">
            <option value="default">Default Mode ({config["retrieval_mode"]})</option>
            <option value="day">Whole Day</option>
            <option value="memory">Memories (#)</option>
            <option value="section">Sections (##)</option>
            <option value="line">Lines (&gt;)</option>
          </select>
          <button onclick="runQuery()">Search</button>
          <div id="results"></div>
        </div>
        <script>
          function saveSettings() {{
            const data = {{
              timezone: document.getElementById('tz').value,
              include_titles: document.getElementById('include_titles').value === 'true',
              retrieval_mode: document.getElementById('mode').value,
              recency_weight: parseFloat(document.getElementById('recency').value) || 0.0,
              n_candidates: parseInt(document.getElementById('n_candidates').value) || 10,
              n_results: parseInt(document.getElementById('n_results').value) || 5
            }};
            fetch('/api/settings', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify(data)
            }})
            .then(res => res.json())
            .then(res => {{
              alert('Settings saved. ' + (res.message || ''));
              location.reload();
            }})
            .catch(err => alert('Error saving settings: ' + err));
          }}

          function refreshIndex() {{
            fetch('/api/refresh', {{ method: 'POST' }})
              .then(res => res.json())
              .then(res => {{
                alert(res.message || 'Index refreshed.');
                location.reload();
              }})
              .catch(err => alert('Error refreshing index: ' + err));
          }}

          function confirmReset() {{
            if (confirm('WARNING: This will delete all indexed data. Are you sure you want to proceed?')) {{
              fetch('/api/reset', {{ method: 'POST' }})
                .then(res => res.json())
                .then(res => {{
                  alert(res.message || 'Index reset.');
                  location.reload();
                }})
                .catch(err => alert('Error resetting index: ' + err));
            }}
          }}

          function runQuery() {{
            const query = document.getElementById('query_input').value;
            if (!query) {{ alert('Please enter a query.'); return; }}
            const mode = document.getElementById('query_mode').value;
            let url = '/api/query?q=' + encodeURIComponent(query);
            if (mode && mode !== 'default') {{
              url += '&mode=' + mode;
            }}
            fetch(url)
              .then(res => res.json())
              .then(res => {{
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = '<h3>Results</h3>';
                if (res.results && res.results.length) {{
                  res.results.forEach(item => {{
                    const div = document.createElement('div');
                    div.className = 'result-item';
                    const title = item.title || '(No title)';
                    const metaParts = [];
                    if (item.date) metaParts.push('Date: ' + item.date);
                    if (item.parent_memory) metaParts.push('Memory: ' + item.parent_memory);
                    if (item.parent_section) metaParts.push('Section: ' + item.parent_section);
                    const meta = metaParts.length ? '<p class="result-meta">' + metaParts.join(' | ') + '</p>' : '';
                    div.innerHTML = `<p><strong>${{title}}</strong></p>
                                     <p>${{item.text.replace(/\\n/g, "<br/>")}}</p>
                                     ${{meta}}`;
                    resultsDiv.appendChild(div);
                  }});
                }} else {{
                  resultsDiv.innerHTML += '<p>No results found.</p>';
                }}
              }})
              .catch(err => alert('Error querying: ' + err));
          }}

          // Auto-refresh status every 5 seconds if indexing is in progress
          function checkStatus() {{
            fetch('/api/settings')
              .then(res => res.json())
              .then(res => {{
                if (res.status.indexing.is_indexing) {{
                  location.reload();
                }}
              }});
          }}
          
          if ({indexing_status['is_indexing']}) {{
            setInterval(checkStatus, 5000);
          }}
        </script>
      </body>
    </html>
    """ 