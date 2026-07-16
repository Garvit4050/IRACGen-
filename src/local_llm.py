"""
Local LLM Wrapper
Provides unified interface for Ollama (and optionally HuggingFace).

Key fix: timeout raised to 600 s so long Analysis sections never time out.
Streaming is used for all calls so the OS-level socket stays alive during
generation — prevents ReadTimeout on slow CPUs.
"""

import logging
import requests

logger = logging.getLogger(__name__)

# ── Timeout constants ─────────────────────────────────────────────────────────
CONNECT_TIMEOUT = 10     # seconds to establish connection
READ_TIMEOUT    = 600    # seconds to wait for the full response
#  600 s = 10 minutes — more than enough even for 1 300-token Analysis on CPU

class LocalLLM:
    """Wrapper for Ollama local LLM."""

    def __init__(self, provider: str = "ollama", model_name: str = "mistral"):
        self.provider   = provider
        self.model_name = model_name

        if provider == "ollama":
            self.base_url = "http://localhost:11434"
            self._check_ollama()

        logger.info(f"✓ Local LLM initialized ({provider}/{model_name})")

    # ── Health check ─────────────────────────────────────────────────────────

    def _check_ollama(self):
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                logger.info("  Ollama is running")
            else:
                logger.warning("  Ollama responded with unexpected status")
        except requests.exceptions.RequestException:
            logger.warning(
                "  Ollama is NOT running! Start it with:  ollama serve"
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt:      Input text
            max_tokens:  Maximum tokens to generate
            temperature: 0 = deterministic, 1 = creative

        Returns:
            Generated string
        """
        if self.provider == "ollama":
            return self._generate_ollama(prompt, max_tokens, temperature)
        raise ValueError(f"Unsupported provider: {self.provider}")

    def chat(
        self,
        messages: list,
        max_tokens: int = 1000,
        temperature: float = 0.3,
    ) -> str:
        """
        Chat-completion style generation.

        Args:
            messages: [{"role": "user"/"assistant", "content": "..."}]

        Returns:
            Assistant reply string
        """
        if self.provider == "ollama":
            return self._chat_ollama(messages, max_tokens, temperature)
        raise ValueError(f"Unsupported provider: {self.provider}")

    # ── Ollama back-end ───────────────────────────────────────────────────────

    def _generate_ollama(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        """
        Call Ollama /api/generate with streaming so the TCP connection
        stays alive during long generations.
        """
        url     = f"{self.base_url}/api/generate"
        payload = {
            "model":  self.model_name,
            "prompt": prompt,
            "stream": True,          # ← streaming keeps socket alive
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            full_text = []
            with requests.post(
                url,
                json=payload,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    import json
                    chunk = json.loads(line)
                    full_text.append(chunk.get("response", ""))
                    if chunk.get("done", False):
                        break

            return "".join(full_text)

        except requests.exceptions.ReadTimeout:
            logger.error(
                "Ollama timed out. "
                f"Current limit is {READ_TIMEOUT}s. "
                "Increase READ_TIMEOUT at the top of local_llm.py "
                "or reduce max_tokens / word_limits in config.yaml."
            )
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise

    def _chat_ollama(
        self, messages: list, max_tokens: int, temperature: float
    ) -> str:
        """Call Ollama /api/chat with streaming."""
        url     = f"{self.base_url}/api/chat"
        payload = {
            "model":    self.model_name,
            "messages": messages,
            "stream":   True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            full_text = []
            with requests.post(
                url,
                json=payload,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    import json
                    chunk = json.loads(line)
                    full_text.append(
                        chunk.get("message", {}).get("content", "")
                    )
                    if chunk.get("done", False):
                        break

            return "".join(full_text)

        except requests.exceptions.ReadTimeout:
            logger.error(
                f"Ollama chat timed out after {READ_TIMEOUT}s."
            )
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama chat request failed: {e}")
            raise

# ── Quick smoke-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    llm = LocalLLM(provider="ollama", model_name="mistral")
    reply = llm.generate(
        "Explain the employment at-will doctrine in 2 sentences.",
        max_tokens=120,
    )
    print("Response:", reply)
