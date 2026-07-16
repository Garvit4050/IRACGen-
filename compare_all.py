"""
compare_all.py - FINAL FIX
Handles BOTH citation formats:
1. Full: Case Name, Citation (Year)
2. Short: Case Name (Year) ← LegalRAG uses this!
"""

import os, sys, re, yaml, logging, argparse, traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR  = Path("results/outputs")
METRICS_DIR = Path("results/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def load_known_cases(kb_path: str) -> set:
    """Build set of known case names."""
    known = set()
    try:
        for txt in Path(kb_path).rglob("*.txt"):
            with open(txt, encoding="utf-8", errors='ignore') as f:
                for line in f:
                    if line.startswith("Case Name:"):
                        case_name = line.replace("Case Name:", "").strip()
                        if case_name:
                            known.add(case_name.lower())
        logger.info(f"Loaded {len(known)} known cases from knowledge base")
        return known
    except Exception as e:
        logger.error(f"Error loading knowledge base: {e}")
        return set()

def extract_citations(text: str):
    """
    Extract citations - handles BOTH formats:
    1. Full: Case Name, Citation Number (Year)
    2. Short: Case Name (Year) ← LegalRAG uses this format!
    
    Returns list of (full_citation_string, case_name_part) tuples.
    """
    citations = []
    
        pattern_full = (
        r'([A-Z][A-Za-z\s\.&,\'-]+?'                     
        r'(?:Inc\.?|LLC|Ltd\.?|L\.L\.C\.|Corp\.?|Co\.)?'  
        r'\s+v\.\s+'                                      
        r'[A-Z][A-Za-z\s\.&,\'-]+?'                      
        r'(?:Inc\.?|LLC|Ltd\.?|L\.L\.C\.|Corp\.?|Co\.)?)'  
        r'\s*,\s*'                                        # Comma before citation
        r'([\d]+\s+[A-Za-z][A-Za-z0-9\.]*\s+[\d]+)'     
        r'\s*[\[\(]([\d]{4})[\]\)]'                      
    )
    
        pattern_short = (
        r'([A-Z][A-Za-z\s\.&,\'-]+?'                     
        r'(?:Inc\.?|LLC|Ltd\.?|L\.L\.C\.|Corp\.?|Co\.)?'  
        r'\s+v\.\s+'                                      
        r'[A-Z][A-Za-z\s\.&,\'-]+?'                      
        r'(?:Inc\.?|LLC|Ltd\.?|L\.L\.C\.|Corp\.?|Co\.)?)'  
        r'\s*[\[\(]([\d]{4})[\]\)]'                      
    )
    
    # Try full pattern first
    matches_full = re.findall(pattern_full, text)
    for name, cite, year in matches_full:
        name_clean = name.strip().rstrip(',')
        full = f"{name_clean}, {cite.strip()} ({year})"
        citations.append((full, name_clean))
    
    # Then try short pattern (avoid duplicates)
    matched_names = {c[1].lower() for c in citations}
    matches_short = re.findall(pattern_short, text)
    for name, year in matches_short:
        name_clean = name.strip().rstrip(',')
        if name_clean.lower() not in matched_names:
            full = f"{name_clean} ({year})"
            citations.append((full, name_clean))
            matched_names.add(name_clean.lower())
    
    # Deduplicate
    seen = set()
    unique_results = []
    for full, name in citations:
        if full not in seen:
            seen.add(full)
            unique_results.append((full, name))
    
    return unique_results

def citation_accuracy(text: str, known_cases: set) -> dict:
    """Calculate citation accuracy."""
    citations = extract_citations(text)
    
    if not citations:
        return {
            "total": 0, "correct": 0, "hallucinated": 0, "accuracy": 0.0,
            "hallucinated_list": [], "correct_list": []
        }
    
    correct, hallucinated = [], []
    
    for full, name in citations:
        name_lower = name.lower().strip()
        # Clean up
        name_clean = re.sub(r'\b(corp\.|inc\.|co\.|ltd\.?|motor)\b', '', name_lower, flags=re.IGNORECASE).strip()
        name_clean = re.sub(r'\s+', ' ', name_clean)
        
        # Extract main party name (before " v. ")
        if ' v. ' in name_clean:
            main_party = name_clean.split(' v. ')[0].strip()
        else:
            main_party = name_clean
        
        # Check against known cases
        found = False
        for known in known_cases:
            known_clean = re.sub(r'\b(corp\.|inc\.|co\.|ltd\.?|motor)\b', '', known, flags=re.IGNORECASE).strip()
            known_clean = re.sub(r'\s+', ' ', known_clean)
            
            # Match if main party appears in known case or vice versa
            if main_party in known_clean or known_clean in name_clean:
                found = True
                break
        
        if found:
            correct.append(full)
        else:
            hallucinated.append(full)
    
    total = len(citations)
    acc = len(correct) / total if total > 0 else 0.0
    
    return {
        "total": total,
        "correct": len(correct),
        "hallucinated": len(hallucinated),
        "accuracy": acc,
        "correct_list": correct,
        "hallucinated_list": hallucinated
    }

def irac_compliance(text: str) -> float:
    """Check IRAC structure."""
    t = text.lower()
    score = 0.0
    
    if "issue" in t[:600] or "whether" in t[:600]:
        score += 0.25
    
    if len(extract_citations(text)) >= 2:
        score += 0.25
    
    if any(p in t for p in ["in this case", "here", "applies", "similarly"]):
        score += 0.25
    
    if any(p in t for p in ["therefore", "conclusion", "accordingly", "thus"]):
        score += 0.25
    
    return score

def compute_metrics(text: str, known_cases: set) -> dict:
    """Compute all metrics for a memo."""
    ca = citation_accuracy(text, known_cases)
    
    return {
        "citation_accuracy": ca["accuracy"],
        "irac_compliance": irac_compliance(text),
        "hallucination_rate": ca["hallucinated"] / ca["total"] if ca["total"] > 0 else 0.0,
        "total_citations": ca["total"],
        "correct_citations": ca["correct"],
        "hallucinated_citations": ca["hallucinated"],
        "hallucinated_list": ca["hallucinated_list"],
        "correct_list": ca["correct_list"]
    }

def print_comparison(q_id: int, query: str, zs: dict, nr: dict, lr: dict, known_cases: set):
    """Print comparison table."""
    zs_m = compute_metrics(zs["memo"], known_cases)
    nr_m = compute_metrics(nr["memo"], known_cases)
    lr_m = compute_metrics(lr["memo"]["full_text"], known_cases)
    
    print("\n" + "="*70)
    print(f"  QUERY {q_id}: {query[:65]}...")
    print("="*70)
    print(f"\n  {'METRIC':<35} {'ZERO-SHOT':>12} {'NAIVE RAG':>12} {'LEGALRAG':>12}")
    print("  " + "─"*71)
    
    metrics = [
        ("Citation Accuracy", "citation_accuracy"),
        ("IRAC Compliance", "irac_compliance"),
        ("Hallucination Rate", "hallucination_rate"),
        ("Total Citations", "total_citations"),
        ("Correct Citations", "correct_citations"),
        ("Hallucinated", "hallucinated_citations")
    ]
    
    for label, key in metrics:
        zv, nv, lv = zs_m[key], nr_m[key], lr_m[key]
        fmt = lambda v: f"{v:.2%}" if isinstance(v, float) else f"{v:>3}"
        print(f"     {label:<35} {fmt(zv):>12} {fmt(nv):>12} {fmt(lv):>12}")
    
    print("\n  ── HALLUCINATED CITATIONS ──────────────────────────────────────")
    for label, m in [("Zero-Shot", zs_m), ("Naive RAG", nr_m), ("LegalRAG", lr_m)]:
        hlist = m.get("hallucinated_list", [])
        if hlist:
            print(f"  {label} ({len(hlist)} hallucinated):")
            for c in hlist[:5]:  # Show max 5
                print(f"      ✗  {c}")
        else:
            print(f"  {label}: ✓ No hallucinated citations")
    
    return zs_m, nr_m, lr_m

def run_single(query: str, config: dict, known_cases: set, q_id: int = 1, save: bool = True):
    """Run single query through all three approaches."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Query {q_id}: {query[:80]}...")
    logger.info(f"{'='*70}")
    
    try:
        # Import baselines
        from baselines.zero_shot import ZeroShotBaseline
        from baselines.naive_rag import NaiveRAG
        
        # Zero-Shot
        logger.info("  [1/3] Running Zero-Shot...")
        zs_baseline = ZeroShotBaseline(config["llm"])
        zs_result = zs_baseline.generate(query)
        logger.info(f"      ✓ Generated {len(zs_result['memo'].split())} words")
        
        # Naive RAG
        logger.info("  [2/3] Running Naive RAG...")
        nr_baseline = NaiveRAG(
            config["llm"],
            config["retrieval"]["hybrid"],
            config["knowledge_base"]["path"]
        )
        nr_result = nr_baseline.generate(query)
        logger.info(f"      ✓ Generated {len(nr_result['memo'].split())} words")
        
        # LegalRAG
        logger.info("  [3/3] Running LegalRAG...")
        from legal_rag import LegalRAG
        legal_rag = LegalRAG(config_path="config.yaml")
        lr_result = legal_rag.generate_memo(query, verbose=False)
        logger.info(f"      ✓ Generated {len(lr_result['memo']['full_text'].split())} words")
        
        # Compute metrics and print
        zs_m, nr_m, lr_m = print_comparison(q_id, query, zs_result, nr_result, lr_result, known_cases)
        
        # Save outputs
        if save:
            for label, text in [
                (f"zero_shot_q{q_id}", zs_result["memo"]),
                (f"naive_rag_q{q_id}", nr_result["memo"]),
                (f"legalrag_q{q_id}", lr_result["memo"]["full_text"])
            ]:
                path = OUTPUT_DIR / f"{label}.txt"
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"QUERY: {query}\n{'='*60}\n\n{text}")
                logger.info(f"  Saved: {path}")
        
        return {
            "query_id": q_id,
            "query": query,
            "zero_shot": zs_m,
            "naive_rag": nr_m,
            "legalrag": lr_m
        }
    
    except Exception as e:
        logger.error(f"  ✗ Query {q_id} FAILED: {e}")
        logger.error(traceback.format_exc())
        return None

def run_all(config: dict, known_cases: set, queries: list):
    """Run all queries."""
    all_results = []
    
    for i, query in enumerate(queries, 1):
        result = run_single(query, config, known_cases, q_id=i)
        if result:
            all_results.append(result)
    
    if not all_results:
        logger.error("No results collected! All queries failed.")
        return []
    
    # Print summary
    print("\n" + "="*70)
    print("  OVERALL RESULTS SUMMARY")
    print("="*70)
    
    for metric in ["citation_accuracy", "irac_compliance", "hallucination_rate"]:
        zs_vals = [r["zero_shot"][metric] for r in all_results]
        nr_vals = [r["naive_rag"][metric] for r in all_results]
        lr_vals = [r["legalrag"][metric] for r in all_results]
        
        zs_avg = sum(zs_vals) / len(zs_vals)
        nr_avg = sum(nr_vals) / len(nr_vals)
        lr_avg = sum(lr_vals) / len(lr_vals)
        
        print(f"  {metric.replace('_', ' ').title():<35} ZS={zs_avg:.2%}  NR={nr_avg:.2%}  LR={lr_avg:.2%}")
    
    # Save CSV
    try:
        import pandas as pd
        rows = []
        for r in all_results:
            for approach in ["zero_shot", "naive_rag", "legalrag"]:
                row = {
                    "query_id": r["query_id"],
                    "query": r["query"][:80],
                    "approach": approach
                }
                row.update(r[approach])
                row.pop("hallucinated_list", None)
                row.pop("correct_list", None)
                rows.append(row)
        
        csv_path = METRICS_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"\n  ✓ CSV saved → {csv_path}")
    except Exception as e:
        logger.warning(f"Could not save CSV: {e}")
    
    return all_results

def load_queries():
    """Load test queries."""
    try:
        with open("data/test_queries.txt", encoding="utf-8") as f:
            queries = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
        logger.info(f"Loaded {len(queries)} queries from test_queries.txt")
        return queries
    except FileNotFoundError:
        logger.error("data/test_queries.txt not found!")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all three approaches")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="Single query")
    group.add_argument("--all", action="store_true", help="All queries")
    group.add_argument("--quick", action="store_true", help="First 3 queries")
    parser.add_argument("--config", default="config.yaml", help="Config file")
    args = parser.parse_args()
    
    # Load config
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file {args.config} not found!")
        sys.exit(1)
    
    # Load knowledge base
    kb_path = config.get("knowledge_base", {}).get("path", "data/legal_cases")
    known_cases = load_known_cases(kb_path)
    
    if not known_cases:
        logger.warning("No cases found in knowledge base!")
    else:
        logger.info(f"Knowledge base: {len(known_cases)} cases")
    
    # Run
    if args.query:
        run_single(args.query, config, known_cases, q_id=1)
    elif args.quick:
        queries = load_queries()[:3]
        if queries:
            run_all(config, known_cases, queries)
    else:
        queries = load_queries()
        if queries:
            run_all(config, known_cases, queries)
