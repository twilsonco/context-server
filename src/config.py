import os
import json
from zoneinfo import ZoneInfo
from datetime import datetime
try:
    import tzlocal
except ImportError:
    tzlocal = None

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
    "port": 5712,                      # uncommon port
    "faiss_dir": "./faiss_index"       # directory to save/load FAISS index
}

def ensure_directory_structure():
    """Ensure the year/month directory structure exists for the current year."""
    base_dir = DEFAULT_CONFIG["docs_dir"]
    current_year = str(datetime.now().year)
    current_month = datetime.now().strftime("%B")  # Full month name
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create year directory
    year_dir = os.path.join(base_dir, current_year)
    os.makedirs(year_dir, exist_ok=True)
    
    # Create month directory
    month_dir = os.path.join(year_dir, current_month)
    os.makedirs(month_dir, exist_ok=True)
    
    return base_dir

def load_config():
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

    # Ensure directory structure exists
    config["docs_dir"] = ensure_directory_structure()
    os.makedirs(config["faiss_dir"], exist_ok=True)

    return config, TZ

config, TZ = load_config() 