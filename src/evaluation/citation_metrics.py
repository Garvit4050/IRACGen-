"""
Citation Metrics Module
Verifies legal citations for accuracy and correctness
"""

import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class CitationVerifier:
    """
    Verifies legal citations in generated documents
    Checks if cited cases exist and are correctly formatted
    """
    
    def __init__(self, knowledge_base_cases: List[Dict]):
        """
        Initialize citation verifier
        
        Args:
            knowledge_base_cases: List of known cases with names and citations
        """
        self.known_cases = self._build_case_index(knowledge_base_cases)
        logger.info(f"Citation verifier initialized with {len(self.known_cases)} known cases")
    
    def _build_case_index(self, cases: List[Dict]) -> Dict:
        """Build searchable index of known cases"""
        index = {}
        for case in cases:
            case_name = case.get('case_name', '').lower()
            citation = case.get('citation', '')
            if case_name:
                index[case_name] = {
                    'citation': citation,
                    'full_text': case.get('content', '')
                }
        return index
    
    def extract_citations(self, text: str) -> List[Dict]:
        """
        Extract legal citations from text
        
        Matches patterns like:
        - Case Name v. Other Name, 123 U.S. 456 (1999)
        - Case Name, 123 U.S. 456 (1999)
        
        Args:
            text: Generated legal text
            
        Returns:
            List of dicts with citation info
        """
        # Pattern for legal citations
        # Matches: Case Name v. Other Name, Citation (Year)
        pattern = r'([A-Z][A-Za-z\'\s\.&,]+(?:v\.|Corp\.|Inc\.|Co\.)(?:\s+[A-Z][A-Za-z\'\s\.&,]+)?),\s+([\d]+\s+[A-Z][a-z\.]*\s+[\d]+(?:\s+\([\w\.]+\))?)\s+\(([\d]{4})\)'
        
        matches = re.findall(pattern, text)
        
        citations = []
        for match in matches:
            case_name = match[0].strip()
            citation = match[1].strip()
            year = match[2]
            
            citations.append({
                'case_name': case_name,
                'citation': citation,
                'year': year,
                'full_citation': f"{case_name}, {citation} ({year})"
            })
        
        logger.info(f"Extracted {len(citations)} citations from text")
        return citations
    
    def verify_citation_accuracy(self, text: str) -> Dict:
        """
        Verify all citations in text
        
        Args:
            text: Generated legal text with citations
            
        Returns:
            Dict with accuracy metrics
        """
        citations = self.extract_citations(text)
        
        if not citations:
            logger.warning("No citations found in text")
            return {
                'total_citations': 0,
                'correct_citations': 0,
                'incorrect_citations': 0,
                'accuracy': 0.0,
                'details': []
            }
        
        results = []
        correct = 0
        
        for cit in citations:
            case_name_lower = cit['case_name'].lower()
            
            # Check if case exists in knowledge base
            exists = self._case_exists(case_name_lower)
            
            if exists:
                # Verify citation format is correct
                kb_citation = self.known_cases[case_name_lower]['citation']
                citation_correct = self._citations_match(cit['citation'], kb_citation)
                
                if citation_correct:
                    correct += 1
                    results.append({
                        'citation': cit['full_citation'],
                        'status': 'CORRECT',
                        'exists': True,
                        'format_correct': True
                    })
                else:
                    results.append({
                        'citation': cit['full_citation'],
                        'status': 'WRONG_FORMAT',
                        'exists': True,
                        'format_correct': False,
                        'expected': f"{cit['case_name']}, {kb_citation}"
                    })
            else:
                results.append({
                    'citation': cit['full_citation'],
                    'status': 'HALLUCINATED',
                    'exists': False,
                    'format_correct': False
                })
        
        accuracy = correct / len(citations) if citations else 0.0
        
        logger.info(f"Citation accuracy: {accuracy:.2%} ({correct}/{len(citations)})")
        
        return {
            'total_citations': len(citations),
            'correct_citations': correct,
            'incorrect_citations': len(citations) - correct,
            'accuracy': accuracy,
            'details': results
        }
    
    def _case_exists(self, case_name: str) -> bool:
        """Check if case exists in knowledge base"""
        case_name = case_name.lower().strip()
        
        # Direct match
        if case_name in self.known_cases:
            return True
        
        # Partial match (case name contains or is contained)
        for known_case in self.known_cases.keys():
            if case_name in known_case or known_case in case_name:
                return True
        
        return False
    
    def _citations_match(self, generated: str, known: str) -> bool:
        """Check if citations match (allowing minor formatting differences)"""
        # Normalize both citations
        gen_norm = self._normalize_citation(generated)
        known_norm = self._normalize_citation(known)
        
        return gen_norm == known_norm
    
    def _normalize_citation(self, citation: str) -> str:
        """Normalize citation for comparison"""
        # Remove extra whitespace
        citation = ' '.join(citation.split())
        # Convert to lowercase
        citation = citation.lower()
        # Remove periods after abbreviations
        citation = citation.replace('.', '')
        return citation

def compute_citation_metrics(generated_text: str, knowledge_base_cases: List[Dict]) -> Dict:
    """
    Convenience function to compute all citation metrics
    
    Args:
        generated_text: Generated legal memo
        knowledge_base_cases: Known cases
        
    Returns:
        Dict with citation metrics
    """
    verifier = CitationVerifier(knowledge_base_cases)
    return verifier.verify_citation_accuracy(generated_text)

def extract_case_names_from_text(text: str) -> List[str]:
    """
    Extract just the case names (for quick checks)
    
    Args:
        text: Legal text
        
    Returns:
        List of case names
    """
    verifier = CitationVerifier([])  # No KB needed for extraction
    citations = verifier.extract_citations(text)
    return [cit['case_name'] for cit in citations]

# Example usage and testing
if __name__ == "__main__":
    # Test citation extraction
    sample_text = """
    As established in Palsgraf v. Long Island Railroad Co., 248 N.Y. 339 (1928),
    duty is owed only to foreseeable plaintiffs. Similarly, in MacPherson v. Buick 
    Motor Co., 217 N.Y. 382 (1916), the court held that manufacturers owe a duty
    to ultimate consumers. However, the holding in Smith v. Jones, 123 F.2d 456 (1950)
    suggests a different approach.
    """
    
    # Mock knowledge base
    kb = [
        {'case_name': 'Palsgraf v. Long Island Railroad Co.', 'citation': '248 N.Y. 339', 'content': '...'},
        {'case_name': 'MacPherson v. Buick Motor Co.', 'citation': '217 N.Y. 382', 'content': '...'}
    ]
    
    verifier = CitationVerifier(kb)
    results = verifier.verify_citation_accuracy(sample_text)
    
    print("Citation Verification Results:")
    print(f"Total: {results['total_citations']}")
    print(f"Correct: {results['correct_citations']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print("\nDetails:")
    for detail in results['details']:
        print(f"  - {detail['citation']}: {detail['status']}")
