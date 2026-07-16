"""
IRAC Generation Module
Generates legal memoranda in IRAC format (Issue-Rule-Analysis-Conclusion).

CPU-friendly fix: prompts are trimmed to only what the LLM needs,
and max_tokens values are sized to match the word_limits in config.yaml
so generation finishes well within the 600-second timeout in local_llm.py.

Rough rule: 1 word ≈ 1.3 tokens, so:
  issue      200 words →  260 tokens  → we ask for  320
  rule       400 words →  520 tokens  → we ask for  600
  analysis   800 words → 1040 tokens  → we ask for 1100
  conclusion 300 words →  390 tokens  → we ask for  450
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))
from local_llm import LocalLLM

logger = logging.getLogger(__name__)

class IRAGenerator:
    """
    Generates structured legal memoranda using IRAC format.
    No artificial chapters — uses natural legal structure.
    """

    def __init__(self, llm_config: Dict, word_limits: Dict):
        """
        Args:
            llm_config:  From config.yaml → llm
            word_limits: From config.yaml → document → word_limits
        """
        self.word_limits = word_limits or {
            "issue": 200, "rule": 400, "analysis": 800, "conclusion": 300
        }

        # ── Initialise LLM ────────────────────────────────────────────────────
        provider = llm_config.get("provider", "local")
        self.provider    = provider
        self.temperature = llm_config.get("temperature", 0.3)

        if provider == "local":
            self.llm = LocalLLM(
                provider=llm_config.get("local_provider", "ollama"),
                model_name=llm_config.get("local_model", "mistral"),
            )
        elif provider == "anthropic":
            from anthropic import Anthropic
            import os
            self._client   = Anthropic(api_key=os.getenv(
                llm_config.get("api_key_env", "ANTHROPIC_API_KEY")))
            self._model    = llm_config.get("api_model", "claude-3-sonnet-20240229")
            self.llm       = None
        elif provider == "openai":
            from openai import OpenAI
            import os
            self._client   = OpenAI(api_key=os.getenv(
                llm_config.get("api_key_env", "OPENAI_API_KEY")))
            self._model    = llm_config.get("api_model", "gpt-4")
            self.llm       = None
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        logger.info(
            f"✓ IRAC Generator initialized | provider={provider} | "
            f"word_limits={self.word_limits}"
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def generate_memorandum(self, query: str, sources: List[Dict]) -> Dict:
        """
        Run the full IRAC pipeline.

        Returns dict with keys:
            issue, rule, analysis, conclusion, full_text, word_count
        """
        logger.info(f"Generating legal memorandum for: {query[:100]}...")

        issue      = self._generate_issue(query, sources)
        rule       = self._generate_rule(query, sources, issue)
        analysis   = self._generate_analysis(query, sources, issue, rule)
        conclusion = self._generate_conclusion(query, analysis)

        full_text  = self._format_memorandum(issue, rule, analysis, conclusion)

        logger.info("✓ Legal memorandum generated successfully")
        return {
            "issue":      issue,
            "rule":       rule,
            "analysis":   analysis,
            "conclusion": conclusion,
            "full_text":  full_text,
            "word_count": len(full_text.split()),
        }

    # =========================================================================
    # Section generators  (each has its own tight token budget)
    # =========================================================================

    def _generate_issue(self, query: str, sources: List[Dict]) -> str:
        logger.info("  Generating ISSUE section...")
        context = self._fmt_sources(sources[:2])
        wl      = self.word_limits["issue"]

        prompt = (
            f"You are a legal expert writing the ISSUE section of a legal memo.\n\n"
            f"LEGAL QUESTION:\n{query}\n\n"
            f"RELEVANT CONTEXT:\n{context}\n\n"
            f"Write the ISSUE in exactly {wl} words.\n"
            f'Start with: "Whether [question based on facts]"\n'
            f"Use formal legal writing. Be precise.\n\n"
            f"ISSUE:"
        )
        # 1 word ≈ 1.3 tokens; add 20 % headroom
        return self._call_llm(prompt, max_tokens=int(wl * 1.6)).strip()

    def _generate_rule(
        self, query: str, sources: List[Dict], issue: str
    ) -> str:
        logger.info("  Generating RULE section...")
        sources_text = self._fmt_sources_cited(sources[:5])
        wl           = self.word_limits["rule"]

        prompt = (
            f"You are a legal expert writing the RULE section of a legal memo.\n\n"
            f"ISSUE:\n{issue}\n\n"
            f"AVAILABLE CASES (cite ONLY these — never invent a citation):\n"
            f"{sources_text}\n\n"
            f"Write the RULE in exactly {wl} words.\n"
            f"Requirements:\n"
            f"  1. State each applicable legal rule clearly.\n"
            f"  2. Cite every case with EXACT format: Name, Citation (Year).\n"
            f"     Example: Palsgraf v. Long Island Railroad Co., 248 N.Y. 339 (1928)\n"
            f"  3. One sentence per case explaining its holding.\n"
            f"  4. Note any exceptions or limitations.\n\n"
            f"RULE:"
        )
        return self._call_llm(prompt, max_tokens=int(wl * 1.6)).strip()

    def _generate_analysis(
        self, query: str, sources: List[Dict], issue: str, rule: str
    ) -> str:
        logger.info("  Generating ANALYSIS section...")
        sources_text = self._fmt_sources_cited(sources)
        wl           = self.word_limits["analysis"]

        # Keep source excerpts short so the total prompt stays manageable
        prompt = (
            f"You are a legal expert writing the ANALYSIS section of a legal memo.\n\n"
            f"ISSUE:\n{issue}\n\n"
            f"RULE (apply every principle here):\n{rule}\n\n"
            f"PRECEDENTS:\n{sources_text}\n\n"
            f"Write the ANALYSIS in exactly {wl} words.\n"
            f"Requirements:\n"
            f"  1. Apply every rule from the RULE section to the specific facts.\n"
            f"  2. Use phrases: 'Here,', 'In this case,', 'Similarly to [Case],'\n"
            f"  3. Compare and distinguish each precedent case.\n"
            f"  4. Analyse each element of the applicable legal test.\n"
            f"  5. Address the strongest counterargument.\n\n"
            f"ANALYSIS:"
        )
        return self._call_llm(prompt, max_tokens=int(wl * 1.6)).strip()

    def _generate_conclusion(self, query: str, analysis: str) -> str:
        logger.info("  Generating CONCLUSION section...")
        wl = self.word_limits["conclusion"]

        # Only pass the last 600 chars of analysis to keep prompt short
        analysis_excerpt = analysis[-600:] if len(analysis) > 600 else analysis

        prompt = (
            f"You are a legal expert writing the CONCLUSION of a legal memo.\n\n"
            f"ANALYSIS (excerpt):\n{analysis_excerpt}\n\n"
            f"Write the CONCLUSION in exactly {wl} words.\n"
            f"Requirements:\n"
            f"  1. Directly answer the legal question.\n"
            f"  2. Summarise the key analytical points.\n"
            f"  3. State the likely outcome using 'likely', 'probably', "
            f"'the court would find'.\n"
            f"  4. Note any genuine uncertainties.\n\n"
            f"CONCLUSION:"
        )
        return self._call_llm(prompt, max_tokens=int(wl * 1.6)).strip()

    # =========================================================================
        # =========================================================================

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """Route to the correct back-end."""
        if self.provider == "local":
            return self.llm.generate(
                prompt,
                max_tokens=max_tokens,
                temperature=self.temperature,
            )
        elif self.provider == "anthropic":
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        elif self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content

    def _fmt_sources(self, sources: List[Dict]) -> str:
        """Brief source context (no citation header) — used for Issue."""
        parts = []
        for i, src in enumerate(sources, 1):
            text = src.get("content", src.get("text", ""))[:300]
            parts.append(f"[Source {i}]\n{text}...")
        return "\n\n".join(parts)

    def _fmt_sources_cited(self, sources: List[Dict]) -> str:
        """Sources with case name + citation — used for Rule and Analysis."""
        parts = []
        for i, src in enumerate(sources, 1):
            name     = src.get("case_name", f"Case {i}")
            citation = src.get("citation", "citation unavailable")
            # Keep excerpt short to reduce prompt size on CPU
            text     = src.get("content", src.get("text", ""))[:400]
            parts.append(f"Case {i}: {name}, {citation}\n{text}...")
        return "\n\n".join(parts)

    def _format_memorandum(
        self, issue: str, rule: str, analysis: str, conclusion: str
    ) -> str:
        sep  = "=" * 60
        dash = "-" * 60
        return (
            f"LEGAL MEMORANDUM\n\n"
            f"{sep}\n\nISSUE\n{dash}\n\n{issue}\n\n"
            f"{sep}\n\nRULE\n{dash}\n\n{rule}\n\n"
            f"{sep}\n\nANALYSIS\n{dash}\n\n{analysis}\n\n"
            f"{sep}\n\nCONCLUSION\n{dash}\n\n{conclusion}\n\n"
            f"{sep}\n"
        )
