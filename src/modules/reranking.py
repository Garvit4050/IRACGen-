"""
Semantic Reranking Module
Uses cross-encoder to precisely score query-document relevance
"""

import logging
from typing import List, Dict
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class SemanticReranker:
    """
    Reranks retrieved documents using cross-encoder
    Module 3 of the LegalRAG pipeline
    """
    
    def __init__(self, model_name: str, top_k: int = 5):
        """
        Initialize Semantic Reranker
        
        Args:
            model_name: Cross-encoder model name
            top_k: Number of top documents to keep
        """
        self.top_k = top_k
        
        logger.info(f"Initializing Semantic Reranker...")
        logger.info(f"  Model: {model_name}")
        
        # Load cross-encoder model
        self.model = CrossEncoder(model_name)
        
        logger.info(f"✓ Reranker ready (top_k={top_k})")
    
    def rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """
        Rerank documents by relevance to query
        
        Args:
            query: Original query
            documents: Retrieved documents from hybrid retrieval
            
        Returns:
            Top-k best documents
        """
        logger.info(f"Reranking {len(documents)} documents...")
        
        if len(documents) == 0:
            logger.warning("No documents to rerank!")
            return []
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            # Use case name + content for better context
            text = f"{doc['case_name']}: {doc['content'][:1000]}"
            pairs.append([query, text])
        
        # Score with cross-encoder
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for i, doc in enumerate(documents):
            doc['rerank_score'] = float(scores[i])
        
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        # Keep top-k
        top_docs = reranked[:self.top_k]
        
        logger.info(f"✓ Reranked to top {len(top_docs)} documents")
        
        # Log top results
        for i, doc in enumerate(top_docs[:3], 1):
            logger.info(f"  {i}. {doc['case_name']} (score: {doc['rerank_score']:.3f})")
        
        return top_docs

if __name__ == "__main__":
    # Test
    import yaml
    from hybrid_retrieval import HybridRetriever
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Retrieve documents
    retriever = HybridRetriever(
        "data/legal_cases",
        config['retrieval']['hybrid']
    )
    
    test_queries = ["employment at-will termination"]
    documents = retriever.retrieve(test_queries)
    
    # Rerank
    reranker = SemanticReranker(
        config['retrieval']['reranking']['model'],
        top_k=5
    )
    
    reranked = reranker.rerank(
        "Can employer terminate for social media posts?",
        documents
    )
    
    print(f"\nTop {len(reranked)} After Reranking:")
    for i, doc in enumerate(reranked, 1):
        print(f"\n{i}. {doc['case_name']}")
        print(f"   Hybrid Score: {doc['score']:.3f}")
        print(f"   Rerank Score: {doc['rerank_score']:.3f}")
