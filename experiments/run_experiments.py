"""
Experiment Runner
Runs all 25 test queries across all 3 approaches
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baselines.zero_shot import ZeroShotBaseline
from baselines.naive_rag import NaiveRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_test_queries(filepath: str) -> list:
    """Load test queries from file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Filter out comments and empty lines
    queries = [line.strip() for line in lines 
               if line.strip() and not line.strip().startswith('#')]
    
    return queries

def run_experiments(config_path: str = "config.yaml", quick_test: bool = False):
    """
    Run full experiments
    
    Args:
        config_path: Path to config file
        quick_test: If True, run only 3 queries for testing
    """
    logger.info("=" * 70)
    logger.info("LEGAL RAG EXPERIMENTS")
    logger.info("=" * 70)
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test queries
    queries = load_test_queries("data/test_queries.txt")
    
    if quick_test:
        queries = queries[:3]
        logger.info(f"QUICK TEST MODE: Running {len(queries)} queries")
    else:
        logger.info(f"FULL TEST: Running {len(queries)} queries")
    
    # Initialize systems
    logger.info("\nInitializing systems...")
    
    zero_shot = ZeroShotBaseline(config['llm'])
    naive_rag = NaiveRAG(
        config['llm'],
        config['retrieval']['hybrid'],
        config['knowledge_base']['path']
    )
    
        logger.info("\n⚠️  For Modular RAG, use: python legal_rag.py --query 'Your question'")
    
    # Results storage
    results = []
    
    # Run experiments
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING EXPERIMENTS")
    logger.info("=" * 70)
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\n[{i}/{len(queries)}] Query: {query[:80]}...")
        
        # Zero-Shot
        try:
            logger.info("  Running Zero-Shot...")
            zero_result = zero_shot.generate(query)
            
            # Save output
            output_path = f"results/outputs/zero_shot_query{i}.txt"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(f"QUERY: {query}\n\n")
                f.write(zero_result['memo'])
            
            results.append({
                'query_id': i,
                'query': query,
                'approach': 'zero_shot',
                'word_count': zero_result['word_count'],
                'output_file': output_path
            })
            
            logger.info(f"    ✓ Generated ({zero_result['word_count']} words)")
            
        except Exception as e:
            logger.error(f"    ✗ Zero-Shot failed: {e}")
        
        # Naive RAG
        try:
            logger.info("  Running Naive RAG...")
            naive_result = naive_rag.generate(query)
            
            # Save output
            output_path = f"results/outputs/naive_rag_query{i}.txt"
            with open(output_path, 'w') as f:
                f.write(f"QUERY: {query}\n\n")
                f.write(naive_result['memo'])
            
            results.append({
                'query_id': i,
                'query': query,
                'approach': 'naive_rag',
                'word_count': naive_result['word_count'],
                'sources_used': naive_result['sources_used'],
                'output_file': output_path
            })
            
            logger.info(f"    ✓ Generated ({naive_result['word_count']} words, {naive_result['sources_used']} sources)")
            
        except Exception as e:
            logger.error(f"    ✗ Naive RAG failed: {e}")
    
    # Save results
    df = pd.DataFrame(results)
    results_path = f"results/metrics/experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.to_csv(results_path, index=False)
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENTS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Outputs saved to: results/outputs/")
    logger.info(f"\nTotal documents generated: {len(results)}")
    logger.info(f"  Zero-Shot: {len([r for r in results if r['approach'] == 'zero_shot'])}")
    logger.info(f"  Naive RAG: {len([r for r in results if r['approach'] == 'naive_rag'])}")
    
    logger.info("\nNext steps:")
    logger.info("  1. Run Modular RAG using: python legal_rag.py --query 'Your question'")
    logger.info("  2. Manually verify citations for sample of outputs")
    logger.info("  3. Compute metrics using evaluation scripts")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Legal RAG experiments")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--quick-test", action="store_true", help="Run only 3 queries for testing")
    parser.add_argument("--full", action="store_true", help="Run all 25 queries")
    
    args = parser.parse_args()
    
    # Default to quick test if neither specified
    quick = args.quick_test or not args.full
    
    run_experiments(args.config, quick_test=quick)
