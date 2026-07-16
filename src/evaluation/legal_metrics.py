"""
Legal Metrics Module
Computes legal-specific evaluation metrics
"""

import logging
from typing import Dict, List
import re

logger = logging.getLogger(__name__)

def compute_irac_compliance(document: str) -> float:
    """
    Compute IRAC structure compliance score
    
    Args:
        document: Generated legal memo
        
    Returns:
        Score 0-1 (1 = perfect IRAC compliance)
    """
    text_lower = document.lower()
    
    # Check for each IRAC component
    has_issue = _check_issue_section(document, text_lower)
    has_rule = _check_rule_section(document, text_lower)
    has_analysis = _check_analysis_section(document, text_lower)
    has_conclusion = _check_conclusion_section(document, text_lower)
    
    # Calculate compliance score
    score = sum([has_issue, has_rule, has_analysis, has_conclusion]) / 4.0
    
    return score

def _check_issue_section(text: str, text_lower: str) -> bool:
    """Check for Issue section"""
    # Look for "ISSUE" header or "Whether" construction
    has_header = 'issue' in text_lower[:500]
    has_whether = 'whether' in text_lower[:500]
    has_question = '?' in text[:500]
    
    return has_header or has_whether or has_question

def _check_rule_section(text: str, text_lower: str) -> bool:
    """Check for Rule section"""
    # Look for "RULE" header and legal citations
    has_header = 'rule' in text_lower
    
    # Count citations (Case Name, Citation (Year))
    citation_pattern = r'[A-Z][A-Za-z\s\.&,]+v\.[A-Z][A-Za-z\s\.&,]+,\s+[\d]+\s+[A-Z][a-z\.]*\s+[\d]+\s+\([\d]{4}\)'
    citations = re.findall(citation_pattern, text)
    has_citations = len(citations) >= 2
    
    return has_header or has_citations

def _check_analysis_section(text: str, text_lower: str) -> bool:
    """Check for Analysis section"""
    # Look for "ANALYSIS" header or analysis phrases
    has_header = 'analysis' in text_lower
    
    analysis_phrases = [
        'in this case',
        'here,',
        'applies to',
        'similarly',
        'in contrast',
        'however',
        'therefore'
    ]
    
    has_phrases = any(phrase in text_lower for phrase in analysis_phrases)
    
    return has_header or has_phrases

def _check_conclusion_section(text: str, text_lower: str) -> bool:
    """Check for Conclusion section"""
    # Look for "CONCLUSION" header or conclusion phrases
    has_header = 'conclusion' in text_lower
    
    conclusion_phrases = [
        'therefore',
        'thus',
        'in conclusion',
        'accordingly',
        'based on the foregoing',
        'it is likely that'
    ]
    
    has_phrases = any(phrase in text_lower for phrase in conclusion_phrases)
    
    return has_header or has_phrases

def compute_precedent_relevance(document: str, sources: List[Dict], query: str) -> float:
    """
    Compute precedent relevance score
    
    Args:
        document: Generated memo
        sources: Source cases used
        query: Original query
        
    Returns:
        Average relevance score 0-1
    """
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Extract citations from document
    citation_pattern = r'([A-Z][A-Za-z\s\.&,]+v\.[A-Z][A-Za-z\s\.&,]+)'
    cited_cases = re.findall(citation_pattern, document)
    
    if not cited_cases:
        return 0.0
    
    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Encode query
    query_emb = model.encode([query])
    
    # For each cited case, find in sources and compute relevance
    relevance_scores = []
    
    for cited_case in cited_cases:
        # Find matching source
        for source in sources:
            if cited_case.lower() in source.get('case_name', '').lower():
                # Compute similarity between query and case content
                case_text = source.get('content', '')[:500]  # First 500 chars
                case_emb = model.encode([case_text])
                
                similarity = cosine_similarity(query_emb, case_emb)[0][0]
                relevance_scores.append(similarity)
                break
    
    if not relevance_scores:
        return 0.0
    
    return float(np.mean(relevance_scores))

def compute_legal_accuracy_simple(document: str, sources: List[Dict]) -> float:
    """
    Simple legal accuracy estimation
    
    Args:
        document: Generated memo
        sources: Source cases
        
    Returns:
        Estimated accuracy 0-1
    """
    # Extract factual claims (sentences with legal terms)
    legal_terms = ['court', 'held', 'holding', 'ruled', 'statute', 'law', 'doctrine']
    
    sentences = [s.strip() for s in document.split('.') if len(s.strip()) > 20]
    legal_sentences = [s for s in sentences if any(term in s.lower() for term in legal_terms)]
    
    if not legal_sentences:
        return 1.0
    
    # Simple heuristic: if citations are present, likely accurate
    # In production, use NLI model
    citations_count = len(re.findall(r'v\.', document))
    legal_sentences_count = len(legal_sentences)
    
    # Rough estimate
    if legal_sentences_count == 0:
        return 1.0
    
    citation_ratio = min(1.0, citations_count / (legal_sentences_count * 0.3))
    
    # Return estimated accuracy
    return 0.7 + (0.3 * citation_ratio)

def compute_all_legal_metrics(document: str, sources: List[Dict], query: str) -> Dict:
    """
    Compute all legal metrics
    
    Args:
        document: Generated memo
        sources: Source cases
        query: Original query
        
    Returns:
        Dict with all metrics
    """
    return {
        'irac_compliance': compute_irac_compliance(document),
        'precedent_relevance': compute_precedent_relevance(document, sources, query),
        'legal_accuracy': compute_legal_accuracy_simple(document, sources)
    }

if __name__ == "__main__":
    # Test
    sample_doc = """
    ISSUE
    Whether an employer can terminate an at-will employee.
    
    RULE
    Employment at-will allows termination for any legal reason. 
    Griggs v. Duke Power Co., 401 U.S. 424 (1971).
    
    ANALYSIS
    In this case, the employee was terminated.
    
    CONCLUSION
    Therefore, the termination is likely lawful.
    """
    
    print("IRAC Compliance:", compute_irac_compliance(sample_doc))
    print("Legal Accuracy:", compute_legal_accuracy_simple(sample_doc, []))
