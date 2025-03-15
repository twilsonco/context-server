import os
import time
import logging
from datetime import datetime
from .markdown_parser import parse_markdown_content, normalize_timestamps

logger = logging.getLogger(__name__)

def index_files(files, vector_store):
    """Index a list of markdown files."""
    if not vector_store:
        raise ValueError("vector_store is required")
    
    for file_path in files:
        if not file_path.endswith(".md"):
            continue

        logger.debug(f"Attempting to index file: {file_path}")
        logger.debug(f"File exists: {os.path.exists(file_path)}")
        logger.debug(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")
        
        # Add a small delay to allow file operations to complete
        time.sleep(0.1)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                logger.debug(f"Successfully opened file: {file_path}")
                content = f.read()
                logger.debug(f"Successfully read file: {file_path} (length: {len(content)})")
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}", exc_info=True)
            continue

        try:
            # Parse the content
            segments = parse_markdown_content(content)
            
            # Get the date from the file path/name
            file_date = None
            try:
                fname = os.path.splitext(os.path.basename(file_path))[0]
                date_str = fname.split(".")[0]  # Get YYYY-MM-DD from filename
                file_date = datetime.strptime(date_str, "%Y-%m-%d")
                date_str = file_date.strftime("%Y-%m-%d")
            except ValueError:
                pass
            
            logger.debug(f"Parsed file {file_path}: {len(segments['day'])} days, {len(segments['memory'])} memories, {len(segments['section'])} sections, {len(segments['line'])} lines")
            
            # Add to vector store
            vector_store.add_segments(segments, file_path, date_str)
            logger.debug(f"Successfully indexed file: {file_path}")
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}", exc_info=True) 