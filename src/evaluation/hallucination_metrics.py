"""
Hallucination Metrics Module
Detects and measures hallucinations in generated text
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

def compute_hallucination_rate(document: str, sources: List[Dict]) -> float:
    """
    Compute hallucination rate
    
    Args:
        document: Generated text
        sources: Source documents
        
    Returns:
        Hallucination rate 0-1 (0 = no hallucinations, 1 = all hallucinated)
    """
    # Extract claims
    claims = extract_claims(document)
    
    if not claims:
        return 0.0
    
    # Check each claim
    hallucinated = 0
    for claim in claims:
        if not is_supported(claim, sources):
            hallucinated += 1
    
    rate = hallucinated / len(claims)
    return rate

def extract_claims(text: str) -> List[str]:
    """Extract factual claims from text"""
    # Split into sentences
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    
    # Filter for factual claims (sentences with verbs like "is", "has", "held")
    factual_verbs = ['is', 'are', 'was', 'were', 'has', 'have', 'held', 'ruled', 'found']
    
    claims = []
    for sentence in sentences:
        if any(verb in sentence.lower().split() for verb in factual_verbs):
            claims.append(sentence)
    
    return claims

def is_supported(claim: str, sources: List[Dict]) -> bool:
    """
    Check if claim is supported by sources
    
    Args:
        claim: Claim to verify
        sources: Source documents
        
    Returns:
        True if supported, False if hallucinated
    """
    # Simple keyword overlap check
    # In production, use NLI model
    
    claim_words = set(claim.lower().split())
    
    for source in sources:
        source_text = source.get('content', '').lower()
        source_words = set(source_text.split())
        
        # Check overlap
        overlap = len(claim_words & source_words)
        if overlap > len(claim_words) * 0.3:  # 30% overlap threshold
            return True
    
    return False

def compute_attribution_score(document: str, sources: List[Dict]) -> float:
    """
    Compute attribution score (what % of text can be traced to sources)
    
    Args:
        document: Generated text
        sources: Source documents
        
    Returns:
        Attribution score 0-1 (1 = all content attributed)
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Extract sentences
    sentences = [s.strip() for s in document.split('.') if len(s.strip()) > 20]
    
    if not sentences:
        return 1.0
    
    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode sentences
    sent_embeddings = model.encode(sentences)
    
    # Encode sources
    source_texts = [s.get('content', '')[:500] for s in sources]
    source_embeddings = model.encode(source_texts)
    
    # For each sentence, check if similar to any source
    attributed = 0
    threshold = 0.5
    
    for sent_emb in sent_embeddings:
        similarities = cosine_similarity([sent_emb], source_embeddings)[0]
        if similarities.max() > threshold:
            attributed += 1
    
    score = attributed / len(sentences)
    return score

def compute_factual_consistency(document: str, sources: List[Dict]) -> float:
    """
    Compute factual consistency using simple heuristics
    
    Args:
        document: Generated text
        sources: Source documents
        
    Returns:
        Consistency score 0-1
    """
    # This is a simplified version
    # In production, use NLI model like BART-MNLI
    
    claims = extract_claims(document)
    
    if not claims:
        return 1.0
    
    supported = sum(1 for claim in claims if is_supported(claim, sources))
    
    score = supported / len(claims)
    return score

def compute_all_hallucination_metrics(document: str, sources: List[Dict]) -> Dict:
    """
    Compute all hallucination-related metrics
    
    Args:
        document: Generated text
        sources: Source documents
        
    Returns:
        Dict with all metrics
    """
    return {
        'hallucination_rate': compute_hallucination_rate(document, sources),
        'attribution_score': compute_attribution_score(document, sources),
        'factual_consistency': compute_factual_consistency(document, sources)
    }

if __name__ == "__main__":
    # Test
    sample_doc = "Employment at-will allows termination. The sky is purple."
    
    sources = [
        {'content': 'Employment at-will doctrine permits termination for any legal reason.'}
    ]
    
    metrics = compute_all_hallucination_metrics(sample_doc, sources)
    print("Hallucination Metrics:", metrics)
