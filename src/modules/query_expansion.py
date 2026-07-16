"""
Query Expansion Module
Expands vague legal queries into specific legal questions
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class QueryExpander:
    """
    Expands single query into multiple specific legal queries
    Module 1 of the LegalRAG pipeline
    """
    
    def __init__(self, llm_config: Dict, num_queries: int = 4):
        """
        Initialize Query Expander
        
        Args:
            llm_config: LLM configuration dict
            num_queries: Number of expanded queries to generate
        """
        self.num_queries = num_queries
        self.llm_config = llm_config
        
        # Initialize LLM client
        self._init_llm(llm_config)
        
        logger.info(f"✓ Query Expander initialized (expand to {num_queries} queries)")
    
    def _init_llm(self, config: Dict):
        """Initialize LLM client based on provider"""
        provider = config.get('provider', 'local')
        
        if provider == 'local':
            from local_llm import LocalLLM
            self.llm = LocalLLM(
                provider=config.get('local_provider', 'ollama'),
                model_name=config.get('local_model', 'mistral')
            )
        elif provider == 'anthropic':
            from anthropic import Anthropic
            import os
            api_key = os.getenv(config.get('api_key_env', 'ANTHROPIC_API_KEY'))
            self.client = Anthropic(api_key=api_key)
            self.model = config.get('api_model', 'claude-3-sonnet-20240229')
        elif provider == 'openai':
            from openai import OpenAI
            import os
            api_key = os.getenv(config.get('api_key_env', 'OPENAI_API_KEY'))
            self.client = OpenAI(api_key=api_key)
            self.model = config.get('api_model', 'gpt-4')
        
        self.provider = provider
    
    def expand(self, query: str) -> List[str]:
        """
        Expand query into multiple specific questions
        
        Args:
            query: Original legal question
            
        Returns:
            List of expanded queries
        """
        logger.info(f"Expanding query: {query[:100]}...")
        
        prompt = self._create_expansion_prompt(query)
        
        # Generate expanded queries
        if self.provider == 'local':
            response = self.llm.generate(prompt, max_tokens=500)
        elif self.provider == 'anthropic':
            message = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            response = message.content[0].text
        elif self.provider == 'openai':
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            response = completion.choices[0].message.content
        
        # Parse response into list of queries
        expanded_queries = self._parse_queries(response)
        
        logger.info(f"✓ Expanded into {len(expanded_queries)} queries")
        
        return expanded_queries
    
    def _create_expansion_prompt(self, query: str) -> str:
        """Create prompt for query expansion"""
        prompt = f"""You are a legal research assistant helping to expand a legal question into specific, targeted queries.

ORIGINAL QUESTION:
{query}

Task: Generate {self.num_queries} specific legal questions that would help research this topic comprehensively. Each question should focus on a different aspect:
1. The main legal doctrine or principle
2. Relevant exceptions or limitations
3. Specific case law or precedents
4. Related legal concepts

Format your response as a numbered list with one question per line.

EXPANDED QUERIES:"""
        
        return prompt
    
    def _parse_queries(self, response: str) -> List[str]:
        """Parse LLM response into list of queries"""
        queries = []
        
        # Split by lines
        lines = response.strip().split('\n')
        
        for line in lines:
            # Remove numbering and whitespace
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Remove common numbering patterns
            for pattern in ['1.', '2.', '3.', '4.', '5.', 
                          '1)', '2)', '3)', '4)', '5)',
                          '-', '•', '*']:
                if line.startswith(pattern):
                    line = line[len(pattern):].strip()
                    break
            
            # Add if non-empty
            if line:
                queries.append(line)
        
        # Ensure we have exactly num_queries (or close to it)
        if len(queries) > self.num_queries:
            queries = queries[:self.num_queries]
        elif len(queries) < self.num_queries:
            # If we got fewer, that's okay, use what we have
            logger.warning(f"Only generated {len(queries)} queries (target: {self.num_queries})")
        
        return queries

# Standalone function for easy use
def expand_query(query: str, llm_config: Dict, num_queries: int = 4) -> List[str]:
    """
    Convenience function to expand a query
    
    Args:
        query: Legal question
        llm_config: LLM configuration
        num_queries: Number of expanded queries
        
    Returns:
        List of expanded queries
    """
    expander = QueryExpander(llm_config, num_queries)
    return expander.expand(query)

if __name__ == "__main__":
    # Test
    import yaml
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    expander = QueryExpander(config['llm'], num_queries=4)
    
    test_query = "Can an employer terminate an employee for social media posts?"
    expanded = expander.expand(test_query)
    
    print("Original Query:")
    print(f"  {test_query}")
    print("\nExpanded Queries:")
    for i, q in enumerate(expanded, 1):
        print(f"  {i}. {q}")
