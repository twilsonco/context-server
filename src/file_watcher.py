import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from .markdown_parser import normalize_timestamps, parse_markdown_content, get_file_date

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MDWatcher:
    def __init__(self, folder: str, vector_store):
        """Initialize the markdown file watcher."""
        self.folder = os.path.abspath(folder)
        self.vector_store = vector_store
        self.observer = Observer()
        
        handler = PatternMatchingEventHandler(
            patterns=["*.md"],
            ignore_directories=True
        )
        
        handler.on_created = self._on_created
        handler.on_modified = self._on_modified
        handler.on_deleted = self._on_deleted
        handler.on_moved = self._on_moved
        
        self.observer.schedule(handler, self.folder, recursive=True)

    def _index_file(self, path: str):
        """Index a markdown file."""
        if not path.endswith(".md"):
            return

        logger.debug(f"Attempting to index file: {path}")
        logger.debug(f"File exists: {os.path.exists(path)}")
        logger.debug(f"File size: {os.path.getsize(path) if os.path.exists(path) else 'N/A'}")
        logger.debug(f"File permissions: {oct(os.stat(path).st_mode)[-3:] if os.path.exists(path) else 'N/A'}")
        logger.debug(f"Current process user: {os.getlogin()}")

        # Add a small delay to allow file operations to complete
        time.sleep(0.1)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                logger.debug(f"Successfully opened file: {path}")
                content = f.read()
                logger.debug(f"Successfully read file: {path} (length: {len(content)})")
        except Exception as e:
            logger.error(f"Could not read {path}: {e}", exc_info=True)
            return

        try:
            # Normalize timestamps and update the file
            normalized_content = normalize_timestamps(content)
            if normalized_content != content:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(normalized_content)
                logger.debug(f"Updated file with normalized timestamps: {path}")

            segments = parse_markdown_content(normalized_content)
            file_date = get_file_date(path)
            date_str = file_date.strftime("%Y-%m-%d") if file_date else None
            logger.debug(f"Parsed file {path}: {len(segments['day'])} days, {len(segments['memory'])} memories, {len(segments['section'])} sections, {len(segments['line'])} lines")
            self.vector_store.add_segments(segments, path, date_str)
            logger.debug(f"Successfully indexed file: {path}")
        except Exception as e:
            logger.error(f"Error processing {path}: {e}", exc_info=True)

    def _on_created(self, event):
        """Handle file creation event."""
        if not event.is_directory:
            logger.debug(f"File created event: {event.src_path}")
            self._index_file(event.src_path)

    def _on_modified(self, event):
        """Handle file modification event."""
        if not event.is_directory:
            logger.debug(f"File modified event: {event.src_path}")
            self._index_file(event.src_path)

    def _on_deleted(self, event):
        """Handle file deletion event."""
        if not event.is_directory:
            logger.debug(f"File deleted event: {event.src_path}")
            self.vector_store.remove_file(event.src_path)

    def _on_moved(self, event):
        """Handle file move/rename event."""
        if not event.is_directory:
            logger.debug(f"File moved/renamed event: {event.src_path} -> {event.dest_path}")
            self.vector_store.remove_file(event.src_path)
            self._index_file(event.dest_path)

    def start(self):
        """Start watching for file changes."""
        logger.info("Starting file watcher")
        self.observer.start()

    def stop(self):
        """Stop watching for file changes."""
        logger.info("Stopping file watcher")
        self.observer.stop()
        self.observer.join()

    def index_all(self):
        """Index all existing markdown files."""
        logger.info(f"Starting full index of directory: {self.folder}")
        
        # Count total files first
        total_files = 0
        for root, dirs, files in os.walk(self.folder):
            for fname in files:
                if fname.endswith(".md"):
                    total_files += 1
        
        # Start indexing session
        self.vector_store.start_indexing(total_files)
        
        try:
            for root, dirs, files in os.walk(self.folder):
                for fname in files:
                    if fname.endswith(".md"):
                        full_path = os.path.join(root, fname)
                        logger.debug(f"Found existing file during startup: {full_path}")
                        self._index_file(full_path)
            self.vector_store.finish_indexing()
        except Exception as e:
            logger.error(f"Error during full indexing: {e}", exc_info=True)
            self.vector_store.finish_indexing(error=str(e))
        
        logger.info("Completed full index") 