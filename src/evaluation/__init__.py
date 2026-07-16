"""
Evaluation package initialization
"""

from .citation_metrics import compute_citation_metrics
from .legal_metrics import compute_all_legal_metrics, compute_irac_compliance
from .hallucination_metrics import compute_all_hallucination_metrics

__all__ = [
    'compute_citation_metrics',
    'compute_all_legal_metrics',
    'compute_irac_compliance',
    'compute_all_hallucination_metrics'
]
