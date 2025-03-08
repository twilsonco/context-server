# FastAPI-Based Continuous Markdown Indexing & Retrieval System

This implementation indexes Markdown files into a persistent FAISS vector database and provides a FastAPI server for semantic search, real-time monitoring, and settings management. The system is designed to work on a local 8GB Mac and supports saving the FAISS index and configuration to disk.

## Features:
1. **File Structure & Monitoring:**
   - Markdown files are stored in a structured folder (e.g., `./notes/{year}/{month}/YYYY-MM-DD.md`).
   - Uses `watchdog` to detect new, updated, or deleted files and re-index them in real time.
2. **Text Processing & Embedding:**
   - Parses the Markdown file into different segments: whole day, memories (`#`), sections (`##`), and individual lines (`>`).
   - Normalizes timestamps (e.g., `startMs=...`) to human-readable format using a configurable timezone (auto-detected initially, with manual override support that persists).
   - Uses `sentence-transformers/all-MiniLM-L6-v2` to embed text.
3. **Vector Database & Persistence:**
   - Embeddings are stored in a local FAISS index with an ID mapping.
   - The FAISS index is saved to disk (and reloaded on startup) for persistence.
4. **Retrieval & Reranking:**
   - Retrieves top candidates based on the chosen segment type.
   - Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-rank results.
   - Applies a recency weight (configurable) to boost recent entries.
5. **FastAPI Server & UI:**
   - Provides API endpoints:
     - `GET /api/settings`: Retrieve current settings and indexing status.
     - `POST /api/settings`: Update settings (timezone, include_titles, mode, recency weight, etc.).
     - `GET /api/query`: Query the index (with optional overrides).
     - `POST /api/refresh`: Re-index all files.
     - `POST /api/reset`: Clear the index.
   - Serves an HTML UI at `/` for viewing status, updating settings, manual refresh, and testing queries.
   - Uses an uncommon port (default 5712) to avoid interfering with local development.
6. **Persistence of Settings & Index:**
   - Settings are stored in `config.json`.
   - FAISS indexes are saved to disk in a folder (default `./faiss_index`) and reloaded on startup.

## Dependencies:
- fastapi, uvicorn, watchdog, sentence-transformers, faiss-cpu, tzlocal, numpy

Setup & Usage Instructions:
Install Dependencies:
bash
Copy
pip install fastapi uvicorn watchdog sentence-transformers faiss-cpu tzlocal numpy
Prepare your Markdown Files: Place your notes in the directory structure ./notes/{year}/{month}/YYYY-MM-DD.md.
Run the Server: Start with:
bash
Copy
uvicorn your_script_name:app --reload --port 5712
Access the UI: Open your browser at http://localhost:5712/ to view status, update settings, refresh/reset the index, and test queries.
API Endpoints:
GET /api/settings to fetch settings and index status.
POST /api/settings to update settings.
GET /api/query?q=... to run a search query.
POST /api/refresh to re-index all files.
POST /api/reset to clear the index.
This complete solution monitors changes, indexes your Markdown notes, and provides persistent search via a local vector database. Enjoy building upon and customizing this system for your needs!

pgsql
Copy

Simply copy the above markdown block (which includes the entire code) into your file, install the dependencies, and run your FastAPI server.

## Full Implementation Code:

```python
import os
import re
import json
import threading
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
try:
    import tzlocal  # for auto timezone detection
except ImportError:
    tzlocal = None

# --- Configuration ---
CONFIG_PATH = "config.json"
DEFAULT_CONFIG = {
    "docs_dir": "./notes",             # directory for markdown files
    "timezone": None,                  # auto-detect if None
    "include_titles": True,            # include section titles in embedding
    "retrieval_mode": "memory",        # default retrieval granularity: "day", "memory", "section", or "line"
    "recency_weight": 0.0,             # recency penalty per day
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "n_candidates": 10,
    "n_results": 5,
    "port": 5712,                    # uncommon port
    "faiss_dir": "./faiss_index"       # directory to save/load FAISS index
}
config = DEFAULT_CONFIG.copy()
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r') as f:
            user_conf = json.load(f)
            config.update(user_conf)
    except Exception as e:
        print(f"Warning: could not load config.json ({e}), using defaults.")
if not config.get("timezone"):
    try:
        if tzlocal:
            config["timezone"] = str(tzlocal.get_localzone())
        else:
            config["timezone"] = "UTC"
    except Exception:
        config["timezone"] = "UTC"
try:
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)
except Exception as e:
    print(f"Could not write config file: {e}")
try:
    TZ = ZoneInfo(config["timezone"])
except Exception as e:
    print(f"Invalid timezone '{config['timezone']}', defaulting to UTC.")
    TZ = ZoneInfo("UTC")
    config["timezone"] = "UTC"

# Ensure FAISS directory exists
os.makedirs(config["faiss_dir"], exist_ok=True)

# --- Initialize Embedding Models ---
embedder = SentenceTransformer(config["embedding_model"])
cross_encoder = CrossEncoder(config["cross_encoder_model"])
vector_dim = embedder.get_sentence_embedding_dimension()

# --- FAISS Index Setup with Persistence ---
def save_faiss_index(index, filename):
    faiss.write_index(index, filename)

def load_faiss_index(filename):
    if os.path.exists(filename):
        return faiss.read_index(filename)
    return None

# Initialize indices for each segment type
faiss_files = {
    'day': os.path.join(config["faiss_dir"], "index_day.faiss"),
    'memory': os.path.join(config["faiss_dir"], "index_memory.faiss"),
    'section': os.path.join(config["faiss_dir"], "index_section.faiss"),
    'line': os.path.join(config["faiss_dir"], "index_line.faiss")
}

index_day     = load_faiss_index(faiss_files['day'])
index_memory  = load_faiss_index(faiss_files['memory'])
index_section = load_faiss_index(faiss_files['section'])
index_line    = load_faiss_index(faiss_files['line'])
# If any index isn't available, create a new one
if index_day is None:
    index_day = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
else:
    index_day = faiss.IndexIDMap(index_day)
if index_memory is None:
    index_memory = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
else:
    index_memory = faiss.IndexIDMap(index_memory)
if index_section is None:
    index_section = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
else:
    index_section = faiss.IndexIDMap(index_section)
if index_line is None:
    index_line = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
else:
    index_line = faiss.IndexIDMap(index_line)

# Data mappings for metadata
file_index_ids = {}    # { file_path: {"day": [ids], "memory": [...], ... } }
id_to_doc = { 'day': {}, 'memory': {}, 'section': {}, 'line': {} }
id_counters = { 'day': 0, 'memory': 0, 'section': 0, 'line': 0 }

index_lock = threading.Lock()

# --- Helper Functions ---
def normalize_timestamps(text: str) -> str:
    pattern = re.compile(r'(startMs\s*[:=]\s*)(\d{13})')
    def _replace(match):
        ms = int(match.group(2))
        dt = datetime.fromtimestamp(ms/1000.0, TZ)
        return match.group(1) + dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    return pattern.sub(_replace, text)

def parse_markdown_content(content: str):
    segments = { 'day': [], 'memory': [], 'section': [], 'line': [] }
    lines = content.splitlines()
    day_lines = []
    for line in lines:
        if line.startswith('# '):
            day_lines.append(line[2:].strip())
        elif line.startswith('## '):
            day_lines.append(line[3:].strip())
        elif line.startswith('>'):
            day_lines.append(line[1:].strip())
        else:
            day_lines.append(line)
    day_text = "\n".join(day_lines).strip()
    if day_text:
        segments['day'].append({ "text": day_text, "title": None })
    current_mem_title = None
    current_mem_lines = []
    current_sec_title = None
    current_sec_lines = []
    for line in lines:
        if line.startswith('# '):
            if current_mem_title is not None:
                if current_sec_title is not None:
                    sec_text = "\n".join(current_sec_lines).strip()
                    if sec_text:
                        text = f"{current_sec_title}\n{sec_text}" if config["include_titles"] else sec_text
                        segments['section'].append({ "text": text, "title": current_sec_title, "file_memory": current_mem_title })
                mem_text = "\n".join(current_mem_lines).strip()
                if mem_text:
                    text = f"{current_mem_title}\n{mem_text}" if config["include_titles"] else mem_text
                    segments['memory'].append({ "text": text, "title": current_mem_title })
            current_mem_title = line[2:].strip()
            current_mem_lines = []
            current_sec_title = None
            current_sec_lines = []
        elif line.startswith('## '):
            if current_sec_title is not None:
                sec_text = "\n".join(current_sec_lines).strip()
                if sec_text:
                    text = f"{current_sec_title}\n{sec_text}" if config["include_titles"] else sec_text
                    segments['section'].append({ "text": text, "title": current_sec_title, "file_memory": current_mem_title })
            current_sec_title = line[3:].strip()
            current_sec_lines = []
        elif line.startswith('>'):
            line_text = line[1:].strip()
            if line_text:
                segments['line'].append({ "text": line_text, "title": None, "file_memory": current_mem_title, "file_section": current_sec_title })
            if current_sec_title is not None:
                current_sec_lines.append(line_text)
            elif current_mem_title is not None:
                current_mem_lines.append(line_text)
            continue
        if current_sec_title is not None:
            if not line.startswith('## '):
                if not line.startswith('>'):
                    current_sec_lines.append(line)
        elif current_mem_title is not None:
            if not line.startswith('# '):
                if not line.startswith('>'):
                    current_mem_lines.append(line)
    if current_mem_title is not None:
        if current_sec_title is not None:
            sec_text = "\n".join(current_sec_lines).strip()
            if sec_text:
                text = f"{current_sec_title}\n{sec_text}" if config["include_titles"] else sec_text
                segments['section'].append({ "text": text, "title": current_sec_title, "file_memory": current_mem_title })
        mem_text = "\n".join(current_mem_lines).strip()
        if mem_text:
            text = f"{current_mem_title}\n{mem_text}" if config["include_titles"] else mem_text
            segments['memory'].append({ "text": text, "title": current_mem_title })
    return segments

def get_file_date(path: str):
    fname = os.path.splitext(os.path.basename(path))[0]
    parts = fname.replace('_', '-').split('-')
    year = month = day = None
    try:
        year = int(parts[0])
        month = int(parts[1])
        day = int(parts[2])
    except:
        try:
            year = int(os.path.basename(os.path.dirname(os.path.dirname(path))))
            month = int(os.path.basename(os.path.dirname(path)))
            day = int(parts[0])
        except:
            return None
    try:
        return datetime(year, month, day)
    except:
        return None

# --- Indexing Functions ---
def index_file(path: str):
    if not path.endswith(".md"):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Could not read {path}: {e}")
        return
    content = normalize_timestamps(content)
    segments = parse_markdown_content(content)
    file_date = get_file_date(path)
    with index_lock:
        if path in file_index_ids:
            for seg_type, ids in file_index_ids[path].items():
                if not ids: 
                    continue
                id_array = faiss.IDSelectorBatch(np.array(ids, dtype='int64'))
                try:
                    if seg_type == 'day':
                        index_day.remove_ids(id_array)
                    elif seg_type == 'memory':
                        index_memory.remove_ids(id_array)
                    elif seg_type == 'section':
                        index_section.remove_ids(id_array)
                    elif seg_type == 'line':
                        index_line.remove_ids(id_array)
                except Exception as err:
                    print(f"Warning: remove_ids failed on {seg_type}: {err}")
                for idx in ids:
                    id_to_doc[seg_type].pop(idx, None)
        else:
            file_index_ids[path] = { 'day': [], 'memory': [], 'section': [], 'line': [] }
        for seg_type, seg_list in segments.items():
            for seg in seg_list:
                text = seg["text"]
                title = seg.get("title")
                vec = embedder.encode(text, normalize_embeddings=True).astype('float32')
                idx = id_counters[seg_type]
                id_counters[seg_type] += 1
                if seg_type == 'day':
                    index_day.add_with_ids(np.array([vec]), np.array([idx], dtype='int64'))
                elif seg_type == 'memory':
                    index_memory.add_with_ids(np.array([vec]), np.array([idx], dtype='int64'))
                elif seg_type == 'section':
                    index_section.add_with_ids(np.array([vec]), np.array([idx], dtype='int64'))
                elif seg_type == 'line':
                    index_line.add_with_ids(np.array([vec]), np.array([idx], dtype='int64'))
                doc_info = {
                    "text": text,
                    "title": title,
                    "file": path,
                    "date": file_date.strftime("%Y-%m-%d") if file_date else None,
                    "type": seg_type
                }
                if seg_type in ('section', 'line'):
                    if seg.get("file_memory"):
                        doc_info["parent_memory"] = seg["file_memory"]
                    if seg.get("file_section"):
                        doc_info["parent_section"] = seg["file_section"]
                id_to_doc[seg_type][idx] = doc_info
                file_index_ids[path].setdefault(seg_type, []).append(idx)
    global last_change_index_time
    last_change_index_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Persist FAISS indices
    save_faiss_index(index_day, faiss_files['day'])
    save_faiss_index(index_memory, faiss_files['memory'])
    save_faiss_index(index_section, faiss_files['section'])
    save_faiss_index(index_line, faiss_files['line'])

def remove_file(path: str):
    if path not in file_index_ids:
        return
    with index_lock:
        for seg_type, ids in file_index_ids[path].items():
            if not ids:
                continue
            try:
                id_array = faiss.IDSelectorBatch(np.array(ids, dtype='int64'))
                if seg_type == 'day':
                    index_day.remove_ids(id_array)
                elif seg_type == 'memory':
                    index_memory.remove_ids(id_array)
                elif seg_type == 'section':
                    index_section.remove_ids(id_array)
                elif seg_type == 'line':
                    index_line.remove_ids(id_array)
            except Exception as err:
                print(f"Warning: remove_ids failed on {seg_type}: {err}")
            for idx in ids:
                id_to_doc[seg_type].pop(idx, None)
        file_index_ids.pop(path, None)
    global last_change_index_time
    last_change_index_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    save_faiss_index(index_day, faiss_files['day'])
    save_faiss_index(index_memory, faiss_files['memory'])
    save_faiss_index(index_section, faiss_files['section'])
    save_faiss_index(index_line, faiss_files['line'])

# --- File Watcher Setup ---
class MDWatcher:
    def __init__(self, folder):
        self.folder = folder
        self.observer = Observer()
        handler = PatternMatchingEventHandler(patterns=["*.md"], ignore_directories=True)
        handler.on_created = lambda event: index_file(event.src_path) if not event.is_directory else None
        handler.on_modified = lambda event: index_file(event.src_path) if not event.is_directory else None
        handler.on_deleted = lambda event: remove_file(event.src_path) if not event.is_directory else None
        handler.on_moved   = lambda event: (remove_file(event.src_path), index_file(event.dest_path)) if not event.is_directory else None
        self.observer.schedule(handler, self.folder, recursive=True)
    def start(self):
        self.observer.start()
    def stop(self):
        self.observer.stop()
        self.observer.join()

# --- FastAPI Setup ---
app = FastAPI(title="Markdown Vector Search", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

watcher = MDWatcher(os.path.abspath(config["docs_dir"]))
last_full_index_time = None
last_change_index_time = None

@app.on_event("startup")
def on_startup():
    global last_full_index_time, last_change_index_time
    os.makedirs(config["docs_dir"], exist_ok=True)
    for root, dirs, files in os.walk(config["docs_dir"]):
        for fname in files:
            if fname.endswith(".md"):
                index_file(os.path.join(root, fname))
    last_full_index_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_change_index_time = last_full_index_time
    watcher.start()

@app.on_event("shutdown")
def on_shutdown():
    watcher.stop()

# --- API Endpoints ---
@app.get("/api/settings", response_class=JSONResponse)
def get_settings():
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
                "day": len(id_to_doc['day']),
                "memory": len(id_to_doc['memory']),
                "section": len(id_to_doc['section']),
                "line": len(id_to_doc['line'])
            }
        }
    }

@app.post("/api/settings", response_class=JSONResponse)
def update_settings(new_settings: dict):
    global TZ
    allowed = {"timezone", "include_titles", "retrieval_mode", "recency_weight", "n_candidates", "n_results"}
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
                    TZ = ZoneInfo(value)
                    config["timezone"] = value
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid timezone: {e}")
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        print(f"Could not save config: {e}")
    return {"settings": config, "message": "Settings updated. Some changes may require an index refresh."}

@app.get("/api/query", response_class=JSONResponse)
def query(q: str, mode: str = None, recency_weight: float = None, n_results: int = None):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query 'q' cannot be empty.")
    query_text = q.strip()
    search_mode = mode or config["retrieval_mode"]
    if search_mode not in {"day", "memory", "section", "line"}:
        raise HTTPException(status_code=400, detail="Invalid mode. Choose from 'day', 'memory', 'section', 'line'.")
    recency = config["recency_weight"] if recency_weight is None else recency_weight
    n_res = config["n_results"] if n_results is None else n_results
    query_vec = embedder.encode(query_text, normalize_embeddings=True).astype('float32')
    with index_lock:
        if search_mode == "day":
            D, I = index_day.search(np.array([query_vec]), config["n_candidates"])
        elif search_mode == "memory":
            D, I = index_memory.search(np.array([query_vec]), config["n_candidates"])
        elif search_mode == "section":
            D, I = index_section.search(np.array([query_vec]), config["n_candidates"])
        else:
            D, I = index_line.search(np.array([query_vec]), config["n_candidates"])
    ids = I[0]
    sims = D[0]
    candidates = []
    for idx, sim in zip(ids, sims):
        if idx == -1:
            continue
        doc = id_to_doc[search_mode].get(int(idx))
        if not doc:
            continue
        candidates.append({ "doc": doc, "score": float(sim) })
    if not candidates:
        return { "query": query_text, "results": [] }
    pairs = [(query_text, c["doc"]["text"]) for c in candidates]
    ce_scores = cross_encoder.predict(pairs)
    for c, ce_score in zip(candidates, ce_scores):
        score = float(ce_score)
        entry_date = None
        if recency and c["doc"].get("date"):
            try:
                entry_date = datetime.fromisoformat(c["doc"]["date"])
            except:
                entry_date = None
        if entry_date:
            days_old = (datetime.now() - entry_date).days
            score = score - recency * days_old
        c["final_score"] = score
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = candidates[:n_res]
    results_out = []
    for item in top_results:
        doc = item["doc"]
        result = {
            "text": doc["text"],
            "date": doc["date"],
            "type": doc["type"],
            "title": doc["title"]
        }
        if doc.get("parent_memory"):
            result["parent_memory"] = doc["parent_memory"]
        if doc.get("parent_section"):
            result["parent_section"] = doc["parent_section"]
        results_out.append(result)
    return { "query": query_text, "mode": search_mode, "results": results_out }

@app.post("/api/refresh", response_class=JSONResponse)
def refresh_index():
    global last_full_index_time, last_change_index_time
    with index_lock:
        file_index_ids.clear()
        for t in id_to_doc:
            id_to_doc[t].clear()
            id_counters[t] = 0
        globals()['index_day'] = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
        globals()['index_memory'] = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
        globals()['index_section'] = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
        globals()['index_line'] = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
        for root, dirs, files in os.walk(config["docs_dir"]):
            for fname in files:
                if fname.endswith(".md"):
                    index_file(os.path.join(root, fname))
    last_full_index_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_change_index_time = last_full_index_time
    return { "message": "Index refreshed successfully.", "last_full_index_time": last_full_index_time }

@app.post("/api/reset", response_class=JSONResponse)
def reset_index():
    global last_full_index_time, last_change_index_time
    with index_lock:
        file_index_ids.clear()
        for t in id_to_doc:
            id_to_doc[t].clear()
            id_counters[t] = 0
        globals()['index_day'] = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
        globals()['index_memory'] = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
        globals()['index_section'] = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
        globals()['index_line'] = faiss.IndexIDMap(faiss.IndexFlatIP(vector_dim))
    last_full_index_time = None
    last_change_index_time = None
    save_faiss_index(index_day, faiss_files['day'])
    save_faiss_index(index_memory, faiss_files['memory'])
    save_faiss_index(index_section, faiss_files['section'])
    save_faiss_index(index_line, faiss_files['line'])
    return { "message": "Index cleared. Use refresh to re-index files." }

@app.get("/", response_class=HTMLResponse)
def serve_ui():
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
        </style>
      </head>
      <body>
        <h1>Markdown Vector Search – Settings</h1>
        <div class="section">
          <h2>Status</h2>
          <p>Real-time indexing: <strong>Active</strong></p>
          <p>Last full index refresh: <strong>{last_full_index_time or "N/A"}</strong></p>
          <p>Last file indexed change: <strong>{last_change_index_time or "N/A"}</strong></p>
          <p>Indexed segments: Day={len(id_to_doc['day'])}, Memory={len(id_to_doc['memory'])}, Section={len(id_to_doc['section'])}, Line={len(id_to_doc['line'])}</p>
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
          <button onclick="resetIndex()">Reset Index</button>
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
          function resetIndex() {{
            fetch('/api/reset', {{ method: 'POST' }})
              .then(res => res.json())
              .then(res => {{
                alert(res.message || 'Index reset.');
                location.reload();
              }})
              .catch(err => alert('Error resetting index: ' + err));
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
        </script>
      </body>
    </html>
    """
# To run the app: uvicorn <this_filename_without_.py>:app --host 0.0.0.0 --port {config["port"]}
