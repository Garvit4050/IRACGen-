<<<<<<< HEAD
# LegalRAG: Modular RAG System for Legal Memorandum Generation

## Overview

LegalRAG is a modular Retrieval-Augmented Generation system designed for automated legal memorandum generation with citation verification. The system implements a 5-module architecture to ensure accurate, well-structured, and verifiable legal documents.

## System Architecture

```
Query → [1] Query Expansion → [2] Hybrid Retrieval → [3] Reranking → [4] IRAC Generation → [5] Verification → Output
```

### Modules

1. **Query Expansion** (`src/modules/query_expansion.py`)
   - Expands single query into 4 specific legal questions
   - Uses local LLM (Mistral via Ollama)

2. **Hybrid Retrieval** (`src/modules/hybrid_retrieval.py`)
   - Dense retrieval: FAISS + all-MiniLM-L6-v2 embeddings
   - Sparse retrieval: BM25
   - Fusion: α=0.5, top_k=20

3. **Semantic Reranking** (`src/modules/reranking.py`)
   - Cross-encoder: ms-marco-MiniLM-L-6-v2
   - Reranks 20 → 5 most relevant cases

4. **IRAC Generation** (`src/modules/irac_generation.py`)
   - Structured prompts for Issue, Rule, Analysis, Conclusion
   - Word limits: Issue(200), Rule(400), Analysis(800), Conclusion(300)

5. **Citation Verification** (`src/modules/verification.py`)
   - Citation extraction via regex
   - Existence verification against knowledge base
   - NLI-based legal accuracy check (BART-MNLI)

## Directory Structure

```
modular_rag_thesis/
├── src/
│   ├── modules/          # 5 core modules
│   ├── baselines/        # Zero-Shot and Naive RAG
│   ├── local_llm.py      # Ollama integration
│   └── utils.py
├── data/
│   ├── legal_cases/      # 40 landmark cases (4 domains)
│   ├── test_queries.txt  # 25 evaluation queries
│   └── processed/        # Embeddings cache
├── results/
│   ├── outputs/          # Generated memos
│   └── metrics/          # Evaluation results
├── legal_rag.py          # Main pipeline
├── compare_all.py        # Evaluation script
├── config.yaml           # Configuration
└── requirements.txt      # Dependencies
```

## Setup Instructions

### Option 1: Local Setup

```bash
# 1. Install Python 3.11+
python --version

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install and start Ollama
# Download from: https://ollama.ai
ollama pull mistral

# 5. Run system
python legal_rag.py
```

### Option 2: Docker Setup

```bash
# 1. Build image
docker-compose build

# 2. Run system
docker-compose up

# 3. View logs
docker-compose logs -f legalrag
```

## Running Experiments

### Quick Test (3 queries)
```bash
python compare_all.py --quick
```

### Full Evaluation (25 queries)
```bash
python compare_all.py --all
```

### Single Query
```bash
python compare_all.py --query "Your legal question here"
```

## Results

Evaluation on 25 queries across 4 legal domains:

| Metric | Zero-Shot | Naive RAG | LegalRAG |
|--------|-----------|-----------|----------|
| Citation Accuracy | 32.8% | 93.9% | **98.5%** |
| IRAC Compliance | 87.5% | 77.1% | **100.0%** |
| Hallucination Rate | 46.3% | 2.0% | **1.5%** |
| Avg Citations/Memo | 2.8 | 5.0 | **9.9** |

**Statistical Significance:** IRAC improvement p < 0.001 (highly significant)

## Knowledge Base

40 landmark cases across 4 domains:
- Employment Law: 12 cases
- Contract Law: 10 cases
- Tort Law: 10 cases
- Criminal Law: 8 cases

All cases include:
- Full case name and citation
- Facts, Issue, Holding, Reasoning
- Legal principles and key quotes

## Configuration

Edit `config.yaml` to modify:
- LLM model and parameters
- Retrieval settings (α, top_k)
- Word limits per IRAC section
- Verification thresholds

## Key Files

- `legal_rag.py` - Main system orchestration
- `compare_all.py` - Comparative evaluation
- `statistical_tests.py` - T-tests and effect sizes
- `config.yaml` - System configuration

## Performance Notes

- Runtime: ~15 minutes per query (local CPU)
- Memory: ~4GB RAM minimum
- GPU: Optional (speeds up embeddings)
- Ollama: Required for local LLM

## Citation

```
Author: [Your Name]
Title: LegalRAG: Modular RAG System for Legal Memorandum Generation
Institution: [Your University]
Year: 2026
```

## License

Academic use only. Not for commercial deployment.

## Contact

For questions regarding this implementation, please contact the author.
=======
# IRACGen-
A Modular RAG System for Legal Memoranda
>>>>>>> ef2aa1121bda5fb814022546cdc2f986d28c45bf
