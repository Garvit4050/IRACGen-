# LegalRAG Setup Guide for Code Review

## Quick Start (Docker - Recommended)

**Prerequisite:** Docker Desktop installed

```bash
# 1. Navigate to project directory
cd /path/to/modular_rag_thesis

# 2. Build and run
docker-compose up

# This will:
# - Build the LegalRAG container
# - Start Ollama for local LLM
# - Run a quick test (3 queries)
# - Results saved to results/outputs/
```

**First run takes ~10 minutes to download models.**

---

## Manual Setup (Windows)

### Step 1: Install Prerequisites

1. **Python 3.11+**
   ```bash
   python --version  # Should be 3.11 or higher
   ```

2. **Ollama (Local LLM)**
   - Download: https://ollama.ai/download
   - Install and verify:
     ```bash
     ollama --version
     ```

### Step 2: Setup Virtual Environment

```bash
# Navigate to project
cd D:\LLM\modular_rag_thesis

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Verify activation (should see (venv) in prompt)
```

### Step 3: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# This takes ~5 minutes
```

### Step 4: Download LLM Model

```bash
# Start Ollama (in separate terminal)
ollama serve

# Pull Mistral model (in project terminal)
ollama pull mistral

# This downloads ~4GB, takes 5-10 minutes
```

### Step 5: Verify Setup

```bash
# Test LLM connection
python -c "import requests; print(requests.get('http://localhost:11434/api/tags').json())"

# Should show Mistral model listed
```

---

## Running the Code

### 1. Quick Test (3 queries, ~15 minutes)

```bash
python compare_all.py --quick
```

**Output:**
- `results/outputs/` - 9 text files (3 queries × 3 approaches)
- `results/metrics/` - CSV with results
- Console output - Comparison tables

### 2. Full Evaluation (25 queries, ~2-3 hours)

```bash
python compare_all.py --all
```

### 3. Single Query Test

```bash
python compare_all.py --query "Can an employer be held liable for harassment?"
```

### 4. Generate Just LegalRAG Memo

```python
from legal_rag import LegalRAG

system = LegalRAG(config_path="config.yaml")
result = system.generate_memo("Your legal question here")
print(result["memo"]["full_text"])
```

---

## Understanding the Output

### Console Output

```
======================================================================
Query 1: Can an employer be held liable...
======================================================================

METRIC                      ZERO-SHOT    NAIVE RAG     LEGALRAG
───────────────────────────────────────────────────────────────
Citation Accuracy              33.33%      100.00%       100.00%
IRAC Compliance               100.00%       75.00%       100.00%
Hallucination Rate             66.67%        0.00%         0.00%
```

### File Outputs

**results/outputs/**
- `zero_shot_q1.txt` - Zero-shot baseline memo
- `naive_rag_q1.txt` - Naive RAG baseline memo
- `legalrag_q1.txt` - LegalRAG memo (with verification)

**results/metrics/**
- `comparison_YYYYMMDD_HHMMSS.csv` - All metrics for analysis

---

## Project Structure Explained

```
modular_rag_thesis/
│
├── src/                    # Source code
│   ├── modules/            # 5 core modules
│   │   ├── query_expansion.py
│   │   ├── hybrid_retrieval.py
│   │   ├── reranking.py
│   │   ├── irac_generation.py
│   │   └── verification.py
│   │
│   ├── baselines/          # Comparison baselines
│   │   ├── zero_shot.py
│   │   └── naive_rag.py
│   │
│   └── local_llm.py        # Ollama integration
│
├── data/
│   ├── legal_cases/        # Knowledge base (40 cases)
│   │   ├── employment/
│   │   ├── contract/
│   │   ├── tort/
│   │   └── criminal/
│   │
│   ├── test_queries.txt    # 25 evaluation queries
│   └── processed/          # Cached embeddings
│
├── results/                # Experiment outputs
│   ├── outputs/            # Generated memos
│   └── metrics/            # CSV results
│
├── legal_rag.py            # Main pipeline
├── compare_all.py          # Evaluation script
├── statistical_tests.py    # Statistical analysis
└── config.yaml             # Configuration
```

---

## Code Review Checklist

### Core System (`legal_rag.py`)
- ✓ Orchestrates all 5 modules
- ✓ Clean separation of concerns
- ✓ Error handling for each module

### Module 1: Query Expansion
- ✓ Expands 1 query → 4 specific questions
- ✓ Uses local LLM (Mistral)

### Module 2: Hybrid Retrieval
- ✓ Dense: FAISS + sentence-transformers
- ✓ Sparse: BM25
- ✓ Fusion with α=0.5

### Module 3: Reranking
- ✓ Cross-encoder scores
- ✓ 20 → 5 best cases

### Module 4: IRAC Generation
- ✓ 4 separate prompts (Issue, Rule, Analysis, Conclusion)
- ✓ Word limits enforced

### Module 5: Verification
- ✓ Citation extraction (regex)
- ✓ Existence check (knowledge base)
- ✓ NLI accuracy (BART-MNLI)

### Evaluation (`compare_all.py`)
- ✓ Runs 3 approaches side-by-side
- ✓ Calculates metrics
- ✓ Generates CSV

---

## Common Issues & Solutions

### Issue: "Ollama connection refused"
```bash
# Solution: Start Ollama
ollama serve
```

### Issue: "Model not found"
```bash
# Solution: Pull model
ollama pull mistral
```

### Issue: "ModuleNotFoundError"
```bash
# Solution: Activate venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: "Out of memory"
```bash
# Solution: Reduce batch size in config.yaml
retrieval:
  hybrid:
    top_k: 10  # Reduce from 20
```

---

## Performance Metrics

**On typical laptop (16GB RAM, i7 CPU):**
- Query expansion: ~30 seconds
- Retrieval: ~5 seconds
- Reranking: ~3 seconds
- Generation: ~5 minutes
- Verification: ~2 seconds
- **Total: ~6 minutes per query**

**For 25 queries:** ~2-3 hours

---

## Configuration Options

Edit `config.yaml`:

```yaml
llm:
  provider: "local"
  model: "mistral"
  temperature: 0.3

retrieval:
  hybrid:
    alpha: 0.5       # Dense/sparse weight
    top_k: 20        # Initial retrieval

reranking:
  top_k: 5           # Final case count

generation:
  word_limits:
    issue: 200
    rule: 400
    analysis: 800
    conclusion: 300

verification:
  enabled: true
  nli_threshold: 0.7
```

---

## Testing Individual Modules

```python
# Test Query Expansion
from src.modules.query_expansion import QueryExpander
expander = QueryExpander(config)
queries = expander.expand("Your question")
print(queries)

# Test Retrieval
from src.modules.hybrid_retrieval import HybridRetriever
retriever = HybridRetriever(config)
results = retriever.retrieve("Your question")
print(results)

# Test Verification
from src.modules.verification import CitationVerifier
verifier = CitationVerifier(knowledge_base, config)
result = verifier.verify("Generated memo text")
print(result)
```

---

## Contact & Support

For questions about the code:
- Check README.md
- Review inline documentation
- Run with --help flag: `python legal_rag.py --help`

---

## Expected Results

After running `python compare_all.py --quick`:

```
OVERALL RESULTS SUMMARY
Citation Accuracy     ZS=33%   NR=94%   LR=99%
IRAC Compliance       ZS=88%   NR=77%   LR=100%
Hallucination Rate    ZS=46%   NR=2%    LR=1%
```

**If you see these numbers, the system is working correctly!**
