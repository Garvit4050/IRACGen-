#!/usr/bin/env python3
"""
LegalRAG - Modular RAG for Legal Document Generation
Main system integrating all 5 modules:
  Module 1 - Query Expansion
  Module 2 - Hybrid Retrieval
  Module 3 - Semantic Reranking
  Module 4 - IRAC Generation
  Module 5 - Citation Verification
"""

import os
import sys
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, List

# ── Make sure Python can find the src/ package ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

# ── Module imports (all names must match the actual class names) ──────────────
from modules.query_expansion  import QueryExpander       # Module 1
from modules.hybrid_retrieval import HybridRetriever     # Module 2
from modules.reranking         import SemanticReranker   # Module 3
from modules.irac_generation   import IRAGenerator       # Module 4  ← fixed
from modules.verification      import CitationVerifier   # Module 5

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Main system class
# ─────────────────────────────────────────────────────────────────────────────

class LegalRAG:
    """
    Complete LegalRAG pipeline.
    Loads config, initialises every module, and exposes generate_memo().
    """

    def __init__(self, config_path: str = "config.yaml"):
        logger.info("=" * 60)
        logger.info("Initializing LegalRAG System")
        logger.info("=" * 60)

        self.config = self._load_config(config_path)
        self._init_modules()

        logger.info("✓ LegalRAG System Ready")
        logger.info("=" * 60)

    # ── Startup helpers ───────────────────────────────────────────────────────

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Config loaded from {config_path}")
        return config

    def _init_modules(self):
        logger.info("\nInitializing modules...")

        # Module 1 ─ Query Expansion
        logger.info("  [1/5] Query Expansion")
        self.query_expander = QueryExpander(
            llm_config=self.config["llm"],
            num_queries=self.config["retrieval"]["query_expansion"]["num_queries"]
        )

        # Module 2 ─ Hybrid Retrieval
        logger.info("  [2/5] Hybrid Retrieval")
        self.retriever = HybridRetriever(
            knowledge_base_path=self.config["knowledge_base"]["path"],
            config=self.config["retrieval"]["hybrid"]
        )

        # Module 3 ─ Semantic Reranking
        logger.info("  [3/5] Semantic Reranking")
        self.reranker = SemanticReranker(
            model_name=self.config["retrieval"]["reranking"]["model"],
            top_k=self.config["retrieval"]["reranking"]["top_k"]
        )

        # Module 4 ─ IRAC Generation
        logger.info("  [4/5] IRAC Generation")
        self.generator = IRAGenerator(                    
            llm_config=self.config["llm"],
            word_limits=self.config["document"]["word_limits"]
        )

        # Module 5 ─ Citation Verification
        logger.info("  [5/5] Citation Verification")
        self.verifier = CitationVerifier(
            knowledge_base_cases=self.retriever.get_all_cases(),
            config=self.config["verification"]
        )

        logger.info("✓ All 5 modules initialized")

    # ── Core pipeline ─────────────────────────────────────────────────────────

    def generate_memo(self, query: str, verbose: bool = True) -> Dict:
        """
        Run the full 5-module pipeline and return a legal memorandum.

        Args:
            query:   Legal question (string)
            verbose: Log progress to console

        Returns:
            Dict with keys: query, memo, verification, metadata
        """
        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info(f"QUERY: {query}")
            logger.info("=" * 60)

        # ── Module 1: Query Expansion ────────────────────────────────────────
        if verbose:
            logger.info("\n[MODULE 1] Query Expansion")
        expanded = self.query_expander.expand(query)
        if verbose:
            for i, q in enumerate(expanded, 1):
                logger.info(f"  {i}. {q}")

        # ── Module 2: Hybrid Retrieval ───────────────────────────────────────
        if verbose:
            logger.info("\n[MODULE 2] Hybrid Retrieval")
        retrieved = self.retriever.retrieve(expanded)
        if verbose:
            logger.info(f"  Retrieved {len(retrieved)} documents")

        # ── Module 3: Semantic Reranking ─────────────────────────────────────
        if verbose:
            logger.info("\n[MODULE 3] Semantic Reranking")
        reranked = self.reranker.rerank(query, retrieved)
        if verbose:
            logger.info(f"  Top {len(reranked)} documents selected")
            for i, doc in enumerate(reranked[:3], 1):
                score = doc.get("rerank_score", doc.get("score", 0))
                logger.info(f"  {i}. {doc['case_name']} (score: {score:.3f})")

        # ── Module 4: IRAC Generation ────────────────────────────────────────
        if verbose:
            logger.info("\n[MODULE 4] IRAC Generation")
        memo_result = self.generator.generate_memorandum(query, reranked)
        if verbose:
            logger.info(f"  Generated {memo_result['word_count']} words")
            logger.info("  Structure: Issue → Rule → Analysis → Conclusion")

        # ── Module 5: Citation Verification ─────────────────────────────────
        if verbose:
            logger.info("\n[MODULE 5] Citation Verification")
        verification = self.verifier.verify(memo_result["full_text"])
        if verbose:
            logger.info(f"  Citations found:   {verification['total_citations']}")
            logger.info(f"  Citation accuracy: {verification['accuracy']:.2%}")
            logger.info(f"  Legal accuracy:    {verification['legal_accuracy']:.2%}")

        # ── Compile final result ─────────────────────────────────────────────
        result = {
            "query":            query,
            "expanded_queries": expanded,
            "retrieved_count":  len(retrieved),
            "reranked_count":   len(reranked),
            "memo":             memo_result,
            "verification":     verification,
            "metadata": {
                "word_count":        memo_result["word_count"],
                "citation_accuracy": verification["accuracy"],
                "legal_accuracy":    verification["legal_accuracy"],
                "irac_compliance":   verification.get("irac_compliance", 0.0),
            }
        }

        if verbose:
            logger.info("\n" + "=" * 60)
            logger.info("✓ Legal Memorandum Generated Successfully")
            logger.info("=" * 60)

        return result

    def save_memo(self, result: Dict, output_path: str):
        """Save the full memo text to a file."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result["memo"]["full_text"])
        logger.info(f"✓ Memo saved to {output_path}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LegalRAG – generate verified legal memoranda"
    )
    parser.add_argument(
        "--query",   type=str, required=True,
        help="Legal question to analyse"
    )
    parser.add_argument(
        "--output",  type=str, default=None,
        help="Save memo to this file (default: print to console)"
    )
    parser.add_argument(
        "--config",  type=str, default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )
    parser.add_argument(
        "--quiet",   action="store_true",
        help="Suppress progress messages"
    )

    args = parser.parse_args()

    # Run
    rag    = LegalRAG(config_path=args.config)
    result = rag.generate_memo(args.query, verbose=not args.quiet)

    if args.output:
        rag.save_memo(result, args.output)
        print(f"\n✓ Memo saved to: {args.output}")
    else:
        sep = "=" * 60
        print(f"\n{sep}\nGENERATED LEGAL MEMORANDUM\n{sep}\n")
        print(result["memo"]["full_text"])
        print(f"\n{sep}\nMETRICS\n{sep}")
        print(f"Word Count        : {result['metadata']['word_count']}")
        print(f"Citation Accuracy : {result['metadata']['citation_accuracy']:.2%}")
        print(f"Legal Accuracy    : {result['metadata']['legal_accuracy']:.2%}")
        print(f"IRAC Compliance   : {result['metadata']['irac_compliance']:.2%}")
        print(sep)

if __name__ == "__main__":
    main()
