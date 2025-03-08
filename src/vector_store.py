import os
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer, CrossEncoder
from .config import config

class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer(config["embedding_model"])
        # Enable multi-process encoding
        self.embedder.max_seq_length = 512  # Optimize for typical text length
        self.embedder.parallel_tokenization = True  # Enable parallel tokenization
        
        self.cross_encoder = CrossEncoder(config["cross_encoder_model"])
        self.vector_dim = self.embedder.get_sentence_embedding_dimension()
        
        # Initialize indices
        self.indices = {
            'day': self._load_or_create_index('day'),
            'memory': self._load_or_create_index('memory'),
            'section': self._load_or_create_index('section'),
            'line': self._load_or_create_index('line')
        }
        
        # Data mappings
        self.file_index_ids = {}  # { file_path: {"day": [ids], "memory": [...], ... } }
        self.id_to_doc = {
            'day': {}, 'memory': {}, 'section': {}, 'line': {}
        }
        self.id_counters = {
            'day': 0, 'memory': 0, 'section': 0, 'line': 0
        }
        
        # Progress tracking
        self.indexing_status = {
            'is_indexing': False,
            'current_file': None,
            'total_files': 0,
            'processed_files': 0,
            'start_time': None,
            'end_time': None,
            'error': None
        }

    def _load_or_create_index(self, index_type):
        """Load existing FAISS index or create new one."""
        filename = os.path.join(config["faiss_dir"], f"index_{index_type}.faiss")
        if os.path.exists(filename):
            index = faiss.read_index(filename)
            if isinstance(index, faiss.IndexIDMap):
                return index
            return faiss.IndexIDMap(index)
        return faiss.IndexIDMap(faiss.IndexFlatIP(self.vector_dim))

    def save_indices(self):
        """Save all FAISS indices to disk."""
        for index_type, index in self.indices.items():
            filename = os.path.join(config["faiss_dir"], f"index_{index_type}.faiss")
            faiss.write_index(index, filename)

    def start_indexing(self, total_files):
        """Start an indexing session."""
        self.indexing_status = {
            'is_indexing': True,
            'current_file': None,
            'total_files': total_files,
            'processed_files': 0,
            'start_time': datetime.now(),
            'end_time': None,
            'error': None
        }

    def finish_indexing(self, error=None):
        """Finish an indexing session."""
        self.indexing_status.update({
            'is_indexing': False,
            'current_file': None,
            'end_time': datetime.now(),
            'error': error
        })

    def get_indexing_status(self):
        """Get current indexing status."""
        status = self.indexing_status.copy()
        if status['is_indexing'] and status['total_files'] > 0:
            status['progress'] = (status['processed_files'] / status['total_files']) * 100
        else:
            status['progress'] = 100
        return status

    def remove_file(self, path: str):
        """Remove all segments from a file from the indices."""
        if path not in self.file_index_ids:
            return
        
        for seg_type, ids in self.file_index_ids[path].items():
            if not ids:
                continue
            try:
                id_array = faiss.IDSelectorBatch(np.array(ids, dtype='int64'))
                self.indices[seg_type].remove_ids(id_array)
            except Exception as err:
                print(f"Warning: remove_ids failed on {seg_type}: {err}")
            for idx in ids:
                self.id_to_doc[seg_type].pop(idx, None)
        
        self.file_index_ids.pop(path, None)
        self.save_indices()

    def add_segments(self, segments: dict, file_path: str, file_date: str = None):
        """Add segments from a file to the indices."""
        if self.indexing_status['is_indexing']:
            self.indexing_status['current_file'] = file_path
            self.indexing_status['processed_files'] += 1

        if file_path in self.file_index_ids:
            self.remove_file(file_path)
        
        self.file_index_ids[file_path] = {
            'day': [], 'memory': [], 'section': [], 'line': []
        }

        # Prepare batches for parallel encoding
        for seg_type, seg_list in segments.items():
            if not seg_list:
                continue

            # Prepare texts and metadata
            texts = []
            metas = []
            for seg in seg_list:
                texts.append(seg["text"])
                metas.append({
                    "title": seg.get("title"),
                    "file_memory": seg.get("file_memory"),
                    "file_section": seg.get("file_section")
                })

            # Encode all texts in parallel
            vectors = self.embedder.encode(
                texts,
                batch_size=32,  # Adjust based on your GPU/CPU
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')

            # Add encoded vectors to FAISS
            ids = np.array([self.id_counters[seg_type] + i for i in range(len(texts))], dtype='int64')
            self.indices[seg_type].add_with_ids(vectors, ids)

            # Store document info
            for i, (text, meta) in enumerate(zip(texts, metas)):
                idx = int(ids[i])
                doc_info = {
                    "text": text,
                    "title": meta["title"],
                    "file": file_path,
                    "date": file_date,
                    "type": seg_type
                }

                if seg_type in ('section', 'line'):
                    if meta["file_memory"]:
                        doc_info["parent_memory"] = meta["file_memory"]
                    if meta["file_section"]:
                        doc_info["parent_section"] = meta["file_section"]

                self.id_to_doc[seg_type][idx] = doc_info
                self.file_index_ids[file_path][seg_type].append(idx)

            self.id_counters[seg_type] += len(texts)

        self.save_indices()

    def search(self, query_text: str, mode: str = None, recency_weight: float = None, n_results: int = None):
        """Search for similar segments."""
        search_mode = mode or config["retrieval_mode"]
        if search_mode not in {"day", "memory", "section", "line"}:
            raise ValueError("Invalid mode. Choose from 'day', 'memory', 'section', 'line'.")

        recency = config["recency_weight"] if recency_weight is None else recency_weight
        n_res = config["n_results"] if n_results is None else n_results

        # Encode query in parallel with other queries if any
        query_vec = self.embedder.encode(
            query_text,
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=True
        ).astype('float32')

        D, I = self.indices[search_mode].search(
            np.array([query_vec]), 
            config["n_candidates"]
        )

        ids = I[0]
        sims = D[0]
        candidates = []

        for idx, sim in zip(ids, sims):
            if idx == -1:
                continue
            doc = self.id_to_doc[search_mode].get(int(idx))
            if not doc:
                continue
            candidates.append({"doc": doc, "score": float(sim)})

        if not candidates:
            return []

        # Cross-encoder reranking in parallel
        pairs = [(query_text, c["doc"]["text"]) for c in candidates]
        ce_scores = self.cross_encoder.predict(
            pairs,
            batch_size=32,  # Adjust based on your GPU/CPU
            show_progress_bar=False
        )

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

        results = []
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
            results.append(result)

        return results

    def reset(self):
        """Reset all indices and mappings."""
        for t in self.id_to_doc:
            self.id_to_doc[t].clear()
            self.id_counters[t] = 0
        
        for index_type in self.indices:
            self.indices[index_type] = faiss.IndexIDMap(faiss.IndexFlatIP(self.vector_dim))
        
        self.file_index_ids.clear()
        self.save_indices() 