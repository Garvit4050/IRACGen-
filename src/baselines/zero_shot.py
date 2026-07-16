"""
Zero-Shot Baseline
Direct LLM generation without retrieval
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from local_llm import LocalLLM

logger = logging.getLogger(__name__)

class ZeroShotBaseline:
    """
    Zero-shot generation (no retrieval)
    Baseline 1: Shows the problem we're solving
    """
    
    def __init__(self, llm_config: dict):
        """Initialize zero-shot baseline"""
        logger.info("Initializing Zero-Shot Baseline...")
        
        self.llm = LocalLLM(
            provider=llm_config.get('local_provider', 'ollama'),
            model_name=llm_config.get('local_model', 'mistral')
        )
        
        logger.info("✓ Zero-Shot Baseline ready")
    
    def generate(self, query: str) -> dict:
        """
        Generate legal memo without retrieval
        
        Args:
            query: Legal question
            
        Returns:
            Dict with generated memo
        """
        logger.info(f"Generating zero-shot for: {query[:100]}...")
        
        prompt = f"""You are a legal expert. Write a professional legal memorandum analyzing this question:

QUESTION: {query}

Write a comprehensive legal memorandum in IRAC format (Issue, Rule, Analysis, Conclusion).
Include relevant case citations and legal principles.
The memorandum should be approximately 1500 words.

LEGAL MEMORANDUM:"""
        
        response = self.llm.generate(prompt, max_tokens=2000, temperature=0.3)
        
        return {
            'query': query,
            'memo': response,
            'word_count': len(response.split()),
            'approach': 'zero_shot'
        }

if __name__ == "__main__":
    import yaml
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    baseline = ZeroShotBaseline(config['llm'])
    
    result = baseline.generate("Can an employer fire an employee for social media posts?")
    
    print("ZERO-SHOT GENERATION:")
    print("=" * 60)
    print(result['memo'])
    print("=" * 60)
    print(f"Word count: {result['word_count']}")
