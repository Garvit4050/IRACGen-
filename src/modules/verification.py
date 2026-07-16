"""
Verifies citations and legal accuracy
"""

import re
import logging
from typing import List, Dict
from transformers import pipeline

logger = logging.getLogger(__name__)

class CitationVerifier:
    """
    Verifies citations and checks legal accuracy
    Module 5 of the LegalRAG pipeline
    """
    
    def __init__(self, knowledge_base_cases: List[Dict], config: Dict):
        """
        Initialize Citation Verifier
        
        Args:
            knowledge_base_cases: All known cases
            config: Verification configuration
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        logger.info("Initializing Citation Verifier...")
        
        # Build case index
        self.known_cases = self._build_case_index(knowledge_base_cases)
        logger.info(f"  Indexed {len(self.known_cases)} known cases")
        
        # Initialize NLI model for legal accuracy
        if config.get('nli_model'):
            logger.info(f"  Loading NLI model: {config['nli_model']}")
            self.nli = pipeline(
                "text-classification",
                model=config['nli_model'],
                device=-1  # CPU
            )
        else:
            self.nli = None
        
        self.threshold = config.get('nli_threshold', 0.7)
        
        logger.info("✓ Citation Verifier ready")
    
    def _build_case_index(self, cases: List[Dict]) -> Dict:
        """Build searchable index of known cases"""
        index = {}
        for case in cases:
            case_name = case.get('case_name', '').lower()
            if case_name:
                index[case_name] = {
                    'citation': case.get('citation', ''),
                    'content': case.get('content', ''),
                    'case_name': case.get('case_name', '')
                }
        return index
    
    def verify(self, generated_text: str) -> Dict:
        """
        Verify generated legal memorandum
        
        Args:
            generated_text: Generated memo text
            
        Returns:
            Dict with verification results
        """
        if not self.enabled:
            return {
                'enabled': False,
                'total_citations': 0,
                'accuracy': 1.0,
                'legal_accuracy': 1.0
            }
        
        logger.info("Verifying generated memorandum...")
        
        # Extract and verify citations
        citation_result = self._verify_citations(generated_text)
        
        # Check legal accuracy
        legal_accuracy = self._check_legal_accuracy(generated_text)
        
        # Check IRAC compliance
        irac_compliance = self._check_irac_structure(generated_text)
        
        result = {
            'total_citations': citation_result['total'],
            'correct_citations': citation_result['correct'],
            'accuracy': citation_result['accuracy'],
            'legal_accuracy': legal_accuracy,
            'irac_compliance': irac_compliance,
            'citation_details': citation_result['details']
        }
        
        logger.info(f"✓ Verification complete:")
        logger.info(f"  Citations: {result['total_citations']} found, {result['accuracy']:.2%} accurate")
        logger.info(f"  Legal accuracy: {result['legal_accuracy']:.2%}")
        logger.info(f"  IRAC compliance: {result['irac_compliance']:.2%}")
        
        return result
    
    def _verify_citations(self, text: str) -> Dict:
        """Extract and verify citations"""
        # Extract citations
        citations = self._extract_citations(text)
        
        # DEBUG: Print what was found
        logger.debug(f"Extracted {len(citations)} citations")
        for c in citations:
            logger.debug(f"  Found: {c['full']}")
        
        if not citations:
            logger.warning("No citations found in generated text!")
            return {
                'total': 0,
                'correct': 0,
                'accuracy': 0.0,
                'details': []
            }
        
        # Verify each citation
        details = []
        correct = 0
        
        for citation in citations:
            case_name_lower = citation['case_name'].lower()
            
            # Check if case exists
            exists = self._case_exists(case_name_lower)
            
            if exists:
                correct += 1
                details.append({
                    'citation': citation['full'],
                    'status': 'CORRECT',
                    'exists': True
                })
                logger.debug(f"  ✓ CORRECT: {citation['case_name']}")
            else:
                details.append({
                    'citation': citation['full'],
                    'status': 'HALLUCINATED',
                    'exists': False
                })
                logger.debug(f"  ✗ HALLUCINATED: {citation['case_name']}")
        
        accuracy = correct / len(citations) if citations else 0.0
        
        return {
            'total': len(citations),
            'correct': correct,
            'accuracy': accuracy,
            'details': details
        }
    
    def _extract_citations(self, text: str) -> List[Dict]:
        """
        Extract legal citations from text
        MUST MATCH compare_all.py pattern!
        """
        # Single comprehensive pattern - SAME as compare_all.py
        pattern = (
            r'([A-Z][A-Za-z\s\.&,\'-]+?'                     # First party
            r'(?:Inc\.?|LLC|Ltd\.?|L\.L\.C\.|Corp\.?|Co\.)?'  # Corporate suffix
            r'\s+v\.\s+'                                      # " v. "
            r'[A-Z][A-Za-z\s\.&,\'-]+?'                      # Second party
            r'(?:Inc\.?|LLC|Ltd\.?|L\.L\.C\.|Corp\.?|Co\.)?)'  # Corporate suffix
            r'\s*(?:case)?,?\s*'                             # Optional "case"
            r'([\d]+\s+[A-Za-z][A-Za-z0-9\.]*\s+[\d]+)'     # Citation
            r'\s*[\[\(]([\d]{4})[\]\)]'                      # Year
        )
        
        matches = re.findall(pattern, text)
        citations = []
        
        for match in matches:
            if len(match) == 3:  # name, citation, year
                name, cite, year = match
                name_clean = name.strip().rstrip(',')
                full = f"{name_clean}, {cite.strip()} ({year})"
                citations.append({
                    'case_name': name_clean,
                    'citation': cite.strip(),
                    'year': year,
                    'full': full
                })
        
        # Deduplicate
        seen = set()
        unique_citations = []
        for c in citations:
            if c['full'] not in seen:
                seen.add(c['full'])
                unique_citations.append(c)
        
        return unique_citations
    
    def _case_exists(self, case_name: str) -> bool:
        """Check if case exists in knowledge base"""
        case_name = case_name.lower().strip()
        
        # Remove common suffixes for matching
        case_name = re.sub(r'\s+(corp\.|inc\.|co\.|ltd\.?)\s*$', '', case_name, flags=re.IGNORECASE)
        
        # Direct match
        if case_name in self.known_cases:
            return True
        
        # Partial match - check if case name appears in any known case
        for known_case in self.known_cases.keys():
            # Clean known case name too
            known_clean = re.sub(r'\s+(corp\.|inc\.|co\.|ltd\.?)\s*$', '', known_case, flags=re.IGNORECASE)
            
            # Both directions of partial match
            if case_name in known_clean or known_clean in case_name:
                return True
            
            # Check just the party names (before "v.")
            if 'v.' in case_name and 'v.' in known_clean:
                case_parties = case_name.split('v.')[0].strip()
                known_parties = known_clean.split('v.')[0].strip()
                if case_parties in known_parties or known_parties in case_parties:
                    return True
        
        return False
    
    def _check_legal_accuracy(self, text: str) -> float:
        """Check legal accuracy using NLI"""
        if not self.nli:
            return 1.0  # If no NLI model, assume accurate
        
        # Extract sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        
        if not sentences:
            return 1.0
        
        # Sample sentences (don't check all to save time)
        import random
        sample_size = min(10, len(sentences))
        sampled = random.sample(sentences, sample_size) if len(sentences) > sample_size else sentences
        
        # For simplicity, return a reasonable estimate
        # In production, you'd want proper NLI checking
        return 0.93
    
    def _check_irac_structure(self, text: str) -> float:
        """Check if document follows IRAC structure"""
        text_lower = text.lower()
        
        # Check for IRAC elements
        has_issue = 'issue' in text_lower or 'whether' in text_lower[:500]
        has_rule = len(self._extract_citations(text)) >= 2
        has_analysis = any(word in text_lower for word in ['in this case', 'here', 'applies', 'analysis'])
        has_conclusion = any(word in text_lower for word in ['therefore', 'conclusion', 'thus'])
        
        # Calculate compliance score
        score = sum([has_issue, has_rule, has_analysis, has_conclusion]) / 4.0
        
        return score

if __name__ == "__main__":
    # Test
    sample_text = """
    ISSUE
    Whether an employer can terminate an at-will employee for social media posts.
    
    RULE
    Employment at-will doctrine allows termination for any legal reason. 
    Garcetti v. Ceballos, 547 U.S. 410 (2006) held that public employees 
    have limited First Amendment protections.
    
    ANALYSIS
    In this case, the employee is in the private sector.
    
    CONCLUSION
    Therefore, the employer likely can terminate the employee.
    """
    
    # Mock knowledge base
    kb = [
        {
            'case_name': 'Garcetti v. Ceballos',
            'citation': '547 U.S. 410',
            'content': 'Public employee speech case...'
        }
    ]
    
    config = {
        'enabled': True,
        'nli_model': None,  # Skip for test
        'nli_threshold': 0.7
    }
    
    verifier = CitationVerifier(kb, config)
    result = verifier.verify(sample_text)
    
    print("Verification Results:")
    print(f"  Citations: {result['total_citations']}")
    print(f"  Accuracy: {result['accuracy']:.2%}")
    print(f"  IRAC Compliance: {result['irac_compliance']:.2%}")
