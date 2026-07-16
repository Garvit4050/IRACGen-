"""
Hybrid Retrieval Module
Combines dense (semantic) and sparse (keyword) search
"""

import os
import logging
import pickle
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    Hybrid retrieval using dense + sparse search
    Module 2 of the LegalRAG pipeline
    """
    
    def __init__(self, knowledge_base_path: str, config: Dict):
        """
        Initialize Hybrid Retriever
        
        Args:
            knowledge_base_path: Path to legal cases
            config: Retrieval configuration
        """
        self.kb_path = Path(knowledge_base_path)
        self.config = config
        self.alpha = config.get('alpha', 0.5)
        self.top_k = config.get('top_k', 20)
        
        logger.info("Initializing Hybrid Retriever...")
        
        # Load legal cases
        self.cases = self._load_cases()
        logger.info(f"  Loaded {len(self.cases)} legal cases")
        
        # Initialize dense retrieval
        self._init_dense()
        
        # Initialize sparse retrieval
        self._init_sparse()
        
        logger.info(f"✓ Hybrid Retriever ready (α={self.alpha}, top_k={self.top_k})")
    
    def _load_cases(self) -> List[Dict]:
        """Load all legal cases from knowledge base"""
        cases = []
        
        for area_dir in self.kb_path.iterdir():
            if not area_dir.is_dir():
                continue
            
            for case_file in area_dir.glob("*.txt"):
                try:
                    with open(case_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse case
                    case = self._parse_case(content, case_file.stem)
                    case['area'] = area_dir.name
                    case['filepath'] = str(case_file)
                    cases.append(case)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {case_file}: {e}")
        
        return cases
    
    def _parse_case(self, content: str, filename: str) -> Dict:
        """Parse case file into structured dict"""
        lines = content.split('\n')
        
        case = {
            'filename': filename,
            'content': content,
            'case_name': '',
            'citation': '',
            'court': ''
        }
        
        # Extract metadata
        for line in lines[:10]:  # Check first 10 lines
            if line.startswith('Case Name:'):
                case['case_name'] = line.replace('Case Name:', '').strip()
            elif line.startswith('Citation:'):
                case['citation'] = line.replace('Citation:', '').strip()
            elif line.startswith('Court:'):
                case['court'] = line.replace('Court:', '').strip()
        
        return case
    
    def _init_dense(self):
        """Initialize dense (semantic) retrieval"""
        logger.info("  Initializing dense retrieval (embeddings)...")
        
        model_name = self.config['dense']['model']
        device = self.config['dense'].get('device', 'cpu')
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(model_name, device=device)
        logger.info(f"    Model: {model_name}")
        
        # Generate embeddings for all cases
        texts = [case['content'] for case in self.cases]
        
        cache_path = Path('data/processed/embeddings.pkl')
        if cache_path.exists() and self.config.get('cache_embeddings', True):
            logger.info("    Loading cached embeddings...")
            with open(cache_path, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            logger.info("    Generating embeddings...")
            self.embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Cache embeddings
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
        
        # Build FAISS index
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"    ✓ Dense index built ({len(self.cases)} cases, {embedding_dim}D)")
    
    def _init_sparse(self):
        """Initialize sparse (keyword) retrieval"""
        logger.info("  Initializing sparse retrieval (BM25)...")
        
        # Tokenize documents
        tokenized_docs = []
        for case in self.cases:
            # Simple tokenization (lowercase, split by space)
            tokens = case['content'].lower().split()
            tokenized_docs.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        
        logger.info(f"    ✓ BM25 index built ({len(self.cases)} cases)")
    
    def retrieve(self, queries: List[str]) -> List[Dict]:
        """
        Retrieve relevant cases using hybrid search
        
        Args:
            queries: List of queries (from query expansion)
            
        Returns:
            Top-k most relevant cases
        """
        logger.info(f"Retrieving with {len(queries)} queries...")
        
        all_scores = np.zeros(len(self.cases))
        
        for query in queries:
            # Dense retrieval
            dense_scores = self._dense_search(query)
            
            # Sparse retrieval
            sparse_scores = self._sparse_search(query)
            
            # Normalize scores to [0, 1]
            dense_scores = self._normalize(dense_scores)
            sparse_scores = self._normalize(sparse_scores)
            
            # Hybrid score: α * dense + (1-α) * sparse
            query_scores = self.alpha * dense_scores + (1 - self.alpha) * sparse_scores
            
            # Accumulate scores
            all_scores += query_scores
        
        # Average scores across all queries
        all_scores = all_scores / len(queries)
        
        # Get top-k indices
        top_k_indices = np.argsort(all_scores)[-self.top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_k_indices:
            case = self.cases[idx].copy()
            case['score'] = float(all_scores[idx])
            results.append(case)
        
        logger.info(f"✓ Retrieved {len(results)} cases")
        
        return results
    
    def _dense_search(self, query: str) -> np.ndarray:
        """Dense (semantic) search"""
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            len(self.cases)
        )
        
        # Convert distances to similarity scores (lower distance = higher similarity)
        # Use negative distance as score
        scores = -distances[0]
        
        return scores
    
    def _sparse_search(self, query: str) -> np.ndarray:
        """Sparse (keyword) search"""
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        return scores
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1]"""
        if scores.max() == scores.min():
            return np.ones_like(scores) * 0.5
        
        normalized = (scores - scores.min()) / (scores.max() - scores.min())
        return normalized
    
    def get_all_cases(self) -> List[Dict]:
        """Get all cases (for verification module)"""
        return self.cases

if __name__ == "__main__":
    # Test
    import yaml
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    retriever = HybridRetriever(
        "data/legal_cases",
        config['retrieval']['hybrid']
    )
    
    test_queries = [
        "employment termination rights",
        "at-will employment doctrine",
        "wrongful discharge"
    ]
    
    results = retriever.retrieve(test_queries)
    
    print(f"\nTop {len(results)} Results:")
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. {result['case_name']}")
        print(f"   Citation: {result['citation']}")
        print(f"   Score: {result['score']:.3f}")
