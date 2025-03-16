from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import logging
import asyncio
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

from .config import config, CONFIG_PATH
from .vector_store import VectorStore
from .file_watcher import MDWatcher
from .limitless_api import sync_lifelogs, get_last_fetched_date
from .indexer import index_files

app = FastAPI(
    title="Context Server",
    version="1.0",
    description="""
    An API for semantic search over lifelog entries. This service indexes and searches through lifelog entries,
    supporting different granularity levels (day, memory, section, line) and providing real-time updates through file watching.
    """
)
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

# Global state for background task
sync_task = None

def get_new_files_since(last_indexed_date):
    """Get list of files modified since the last indexed date."""
    new_files = []
    try:
        for root, dirs, files in os.walk(config["docs_dir"]):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if not last_indexed_date or file_mtime > last_indexed_date:
                        new_files.append(file_path)
    except Exception as e:
        print(f"Error getting new files: {e}")
    return new_files

async def periodic_sync():
    """Background task to periodically sync lifelogs."""
    while True:
        try:
            if config.get("limitless_api_key"):
                logger.info("Running periodic sync of lifelogs...")
                sync_lifelogs()
            await asyncio.sleep(config.get("sync_interval_minutes", 3) * 60)
        except Exception as e:
            logger.error(f"Error in periodic sync: {e}")
            await asyncio.sleep(60)  # Wait a minute before retrying on error

@app.on_event("startup")
async def startup_event():
    """Initialize the server and sync/index only new files."""
    global sync_task
    
    if not os.path.exists(config["docs_dir"]):
        os.makedirs(config["docs_dir"])
    
    # Sync new lifelogs if API key is configured
    if config.get("limitless_api_key"):
        sync_lifelogs()
    
    # Only index files that are new or modified
    last_indexed_file = os.path.join(config["docs_dir"], ".last_indexed")
    last_indexed_date = None
    
    try:
        if os.path.exists(last_indexed_file):
            with open(last_indexed_file, 'r') as f:
                timestamp = float(f.read().strip())
                last_indexed_date = datetime.fromtimestamp(timestamp)
    except Exception as e:
        print(f"Error reading last indexed date: {e}")
    
    new_files = get_new_files_since(last_indexed_date)
    if new_files:
        print(f"Indexing {len(new_files)} new or modified files...")
        index_files(new_files, vector_store)
    
    # Update last indexed timestamp
    try:
        with open(last_indexed_file, 'w') as f:
            f.write(str(datetime.now().timestamp()))
    except Exception as e:
        print(f"Error writing last indexed date: {e}")
    
    # Start the file watcher
    watcher.start()
    
    # Start periodic sync task
    sync_task = asyncio.create_task(periodic_sync())

@app.on_event("shutdown")
def on_shutdown():
    """Clean up on application shutdown."""
    logger.info("Application shutting down...")
    if sync_task:
        sync_task.cancel()
    watcher.stop()
    logger.info("Shutdown complete")

@app.get("/api/settings", response_class=JSONResponse)
def get_settings():
    """
    Retrieve current application settings and indexing status.

    Returns a JSON object containing:
    - settings: Current configuration including docs directory, timezone, retrieval preferences
    - status: Indexing status including:
        - Last full/change index times
        - Number of indexed segments by type (day/memory/section/line)
        - Current indexing progress if active
    
    Returns:
        JSONResponse: Current settings and indexing status
    """
    indexing_status = vector_store.get_indexing_status()
    return {
        "settings": {
            "docs_dir": config["docs_dir"],
            "timezone": config["timezone"],
            "include_titles": config["include_titles"],
            "retrieval_mode": config["retrieval_mode"],
            "recency_weight": config["recency_weight"],
            "n_candidates": config["n_candidates"],
            "n_results": config["n_results"],
            "limitless_api_key": config.get("limitless_api_key", ""),
            "sync_interval_minutes": config.get("sync_interval_minutes", 3)
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
    """
    Update application settings.

    Allowed settings:
    - timezone: String, valid timezone name (e.g., 'UTC', 'America/New_York')
    - include_titles: Boolean, whether to include titles in embeddings
    - retrieval_mode: String, default search mode ('day', 'memory', 'section', 'line')
    - recency_weight: Float, weight for recency bias in search results
    - n_candidates: Integer, number of initial candidates for reranking
    - n_results: Integer, number of final results to return
    - limitless_api_key: String, API key for Limitless integration
    - sync_interval_minutes: Integer, sync interval in minutes

    Note: Some changes may require an index refresh to take effect.

    Args:
        new_settings (dict): Dictionary of settings to update

    Returns:
        JSONResponse: Updated settings and confirmation message
    
    Raises:
        HTTPException: If timezone is invalid
    """
    allowed = {
        "timezone", "include_titles", "retrieval_mode",
        "recency_weight", "n_candidates", "n_results",
        "limitless_api_key", "sync_interval_minutes"
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
            elif key == "limitless_api_key":
                # Trigger a sync when API key is updated
                sync_lifelogs(value)
            elif key == "sync_interval_minutes":
                config[key] = int(value)

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
    """
    Search the vector store for semantically similar content.

    The search uses a two-stage retrieval process:
    1. Initial retrieval using sentence embeddings
    2. Reranking using a cross-encoder model

    Args:
        q (str): The search query text
        mode (str, optional): Retrieval granularity ('day', 'memory', 'section', 'line'). 
            Defaults to configured retrieval_mode.
        recency_weight (float, optional): Weight for recency bias. Higher values favor recent content.
            Defaults to configured recency_weight.
        n_results (int, optional): Number of results to return. 
            Defaults to configured n_results.

    Returns:
        JSONResponse: Search results containing:
            - query: Original search query
            - mode: Retrieval mode used
            - results: List of matching segments with metadata

    Raises:
        HTTPException: If query is empty
    """
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
    """
    Perform a full refresh of the search index.

    This operation:
    1. Clears the existing vector store
    2. Re-indexes all markdown files in the configured directory
    3. Updates the last indexing timestamp

    Returns:
        JSONResponse: Confirmation message and timestamp of refresh
    """
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
    """
    Clear the search index completely.

    This operation:
    1. Removes all indexed content from the vector store
    2. Resets indexing timestamps
    3. Does NOT delete any source files

    Note: Use /api/refresh to rebuild the index after resetting.

    Returns:
        JSONResponse: Confirmation message
    """
    global last_full_index_time, last_change_index_time
    vector_store.reset()
    last_full_index_time = None
    last_change_index_time = None
    return {
        "message": "Index cleared. Use refresh to re-index files."
    }

@app.get("/fetch-new", response_class=HTMLResponse)
async def fetch_new():
    """
    Fetch and index new lifelog entries from the Limitless API.

    This endpoint:
    1. Retrieves new entries since the last sync
    2. Saves them as markdown files
    3. Indexes the new content
    4. Updates the last sync timestamp

    Returns:
        HTMLResponse: Success/error message with refresh button

    Raises:
        HTTPException: If Limitless API key is not configured
    """
    if not config.get("limitless_api_key"):
        raise HTTPException(status_code=400, detail="Limitless API key not configured")
    
    try:
        sync_lifelogs()
        # Only index new files
        last_indexed_file = os.path.join(config["docs_dir"], ".last_indexed")
        last_indexed_date = None
        
        try:
            if os.path.exists(last_indexed_file):
                with open(last_indexed_file, 'r') as f:
                    timestamp = float(f.read().strip())
                    last_indexed_date = datetime.fromtimestamp(timestamp)
        except Exception:
            pass
        
        new_files = get_new_files_since(last_indexed_date)
        if new_files:
            index_files(new_files)
            
        # Update last indexed timestamp
        with open(last_indexed_file, 'w') as f:
            f.write(str(datetime.now().timestamp()))
        
        return """
        <div class="alert alert-success">
            Successfully fetched and indexed new entries.
            <button onclick="window.location.reload()" class="btn btn-primary">Refresh Page</button>
        </div>
        """
    except Exception as e:
        return f"""
        <div class="alert alert-danger">
            Error fetching new entries: {str(e)}
            <button onclick="window.location.reload()" class="btn btn-primary">Refresh Page</button>
        </div>
        """

@app.post("/api/refresh-lifelogs", response_class=JSONResponse)
async def refresh_lifelogs():
    """
    Refresh lifelogs from the Limitless API.

    This endpoint:
    1. Syncs all available lifelog entries
    2. Converts them to markdown format
    3. Updates existing files if needed
    4. Returns the sync completion time

    Returns:
        JSONResponse: Success message and sync timestamp

    Raises:
        HTTPException: If Limitless API key is not configured or sync fails
    """
    if not config.get("limitless_api_key"):
        raise HTTPException(status_code=400, detail="Limitless API key not configured")
    
    try:
        sync_lifelogs()
        return {
            "message": "Successfully refreshed lifelogs.",
            "last_sync": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """
    Serve the web UI for managing the search service.

    The UI provides:
    - Current indexing status
    - Configuration settings
    - Search interface
    - Index maintenance operations

    Returns:
        HTMLResponse: HTML page containing the management UI
    """
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
          .button-group {{ margin-top: 1em; }}
          .button-group button {{ margin-right: 0.5em; }}
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
          <div class="field">
            <label>API Key:</label>
            <input type="password" id="api_key" value="{config.get("limitless_api_key", "")}" size="40"/>
          </div>
          <div class="field">
            <label>Sync Interval:</label>
            <input type="number" id="sync_interval" value="{config.get("sync_interval_minutes", 3)}" min="1"/> minutes
          </div>
          <button onclick="saveSettings()">Save Settings</button>
        </div>
        <div class="section">
          <h2>Maintenance</h2>
          <div class="button-group">
            <button onclick="refreshIndex()">Refresh Index</button>
            <button onclick="confirmReset()">Reset Index</button>
            <button onclick="refreshLifelogs()">Refresh Lifelogs</button>
          </div>
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
              n_results: parseInt(document.getElementById('n_results').value) || 5,
              limitless_api_key: document.getElementById('api_key').value,
              sync_interval_minutes: parseInt(document.getElementById('sync_interval').value) || 3
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

          function refreshLifelogs() {{
            fetch('/api/refresh-lifelogs', {{ method: 'POST' }})
              .then(res => res.json())
              .then(res => {{
                alert(res.message || 'Lifelogs refreshed.');
                location.reload();
              }})
              .catch(err => alert('Error refreshing lifelogs: ' + err));
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

# Remove the old settings route and redirect since we're serving the UI at root
@app.get("/settings", response_class=RedirectResponse)
async def settings_redirect():
    """Redirect /settings to root."""
    return RedirectResponse(url="/") 