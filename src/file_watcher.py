import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
from .indexer import index_files

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

    def _on_created(self, event):
        """Handle file creation event."""
        if not event.is_directory:
            logger.debug(f"File created event: {event.src_path}")
            index_files([event.src_path], self.vector_store)

    def _on_modified(self, event):
        """Handle file modification event."""
        if not event.is_directory:
            logger.debug(f"File modified event: {event.src_path}")
            index_files([event.src_path], self.vector_store)

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
            index_files([event.dest_path], self.vector_store)

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
            files_to_index = []
            for root, dirs, files in os.walk(self.folder):
                for fname in files:
                    if fname.endswith(".md"):
                        full_path = os.path.join(root, fname)
                        logger.debug(f"Found existing file during startup: {full_path}")
                        files_to_index.append(full_path)
            
            index_files(files_to_index, self.vector_store)
            self.vector_store.finish_indexing()
        except Exception as e:
            logger.error(f"Error during full indexing: {e}", exc_info=True)
            self.vector_store.finish_indexing(error=str(e))
        
        logger.info("Completed full index") 