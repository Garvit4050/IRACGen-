"""
Naive RAG Baseline
Simple retrieve-then-generate (single-stage)
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.hybrid_retrieval import HybridRetriever
from local_llm import LocalLLM

logger = logging.getLogger(__name__)

class NaiveRAG:
    """
    Naive RAG: Simple retrieve-then-generate
    Baseline 2: Shows value of modular approach
    """
    
    def __init__(self, llm_config: dict, retrieval_config: dict, kb_path: str):
        """Initialize naive RAG"""
        logger.info("Initializing Naive RAG Baseline...")
        
        self.llm = LocalLLM(
            provider=llm_config.get('local_provider', 'ollama'),
            model_name=llm_config.get('local_model', 'mistral')
        )
        
        self.retriever = HybridRetriever(kb_path, retrieval_config)
        
        logger.info("✓ Naive RAG Baseline ready")
    
    def generate(self, query: str) -> dict:
        """
        Generate with simple RAG
        
        Args:
            query: Legal question
            
        Returns:
            Dict with generated memo
        """
        logger.info(f"Generating naive RAG for: {query[:100]}...")
        
        # Simple retrieval (no query expansion, no reranking)
        docs = self.retriever.retrieve([query])
        
        # Take top 5
        top_docs = docs[:5]
        
        # Format sources
        sources_text = self._format_sources(top_docs)
        
        # Generate with sources
        prompt = f"""You are a legal expert. Write a professional legal memorandum analyzing this question:

QUESTION: {query}

RELEVANT LEGAL PRECEDENTS:
{sources_text}

Write a comprehensive legal memorandum in IRAC format using the precedents above.
Cite cases using proper format: Case Name, Citation (Year).
The memorandum should be approximately 1500 words.

LEGAL MEMORANDUM:"""
        
        response = self.llm.generate(prompt, max_tokens=2000, temperature=0.3)
        
        return {
            'query': query,
            'memo': response,
            'word_count': len(response.split()),
            'sources_used': len(top_docs),
            'approach': 'naive_rag'
        }
    
    def _format_sources(self, docs: list) -> str:
        """Format source documents for prompt"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            formatted.append(f"""
Case {i}: {doc['case_name']}, {doc['citation']}
{doc['content'][:500]}...
""")
        return "\n".join(formatted)

if __name__ == "__main__":
    import yaml
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    baseline = NaiveRAG(
        config['llm'],
        config['retrieval']['hybrid'],
        "data/legal_cases"
    )
    
    result = baseline.generate("Can an employer fire an employee for social media posts?")
    
    print("NAIVE RAG GENERATION:")
    print("=" * 60)
    print(result['memo'])
    print("=" * 60)
    print(f"Word count: {result['word_count']}")
    print(f"Sources used: {result['sources_used']}")
