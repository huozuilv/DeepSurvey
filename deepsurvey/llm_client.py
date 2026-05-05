"""
LLM client with chain-of-thought support.
Supports OpenAI-compatible APIs, Anthropic API, and a simulation mode for demo.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CoTResult:
    """Result of a chain-of-thought LLM call."""
    reasoning: str
    answer: str
    raw_response: str = ""
    tokens_used: int = 0


class LLMClient:
    """
    Unified LLM client supporting multiple backends with chain-of-thought.
    Falls back to simulation mode when no API key is configured.
    """

    def __init__(
        self,
        provider: str = "auto",
        model: str = "",
        api_key: str = "",
        base_url: str = "",
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")

        if provider == "auto":
            if os.environ.get("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
                self.model = model or "claude-sonnet-4-6"
            elif self.api_key:
                self.provider = "openai"
                self.model = model or "gpt-4o"
            else:
                self.provider = "simulation"
                self.model = "simulation"

    # ── Public API ───────────────────────────────────────────────────────────

    def chat(self, messages: List[dict], temperature: float = 0.3) -> str:
        """Simple chat completion, returns answer text."""
        result = self._dispatch(messages, temperature, cot=False)
        return result.answer

    def chain_of_thought(
        self,
        messages: List[dict],
        temperature: float = 0.3,
    ) -> CoTResult:
        """Chat completion with explicit chain-of-thought reasoning."""
        return self._dispatch(messages, temperature, cot=True)

    # ── Dispatcher ───────────────────────────────────────────────────────────

    def _dispatch(
        self,
        messages: List[dict],
        temperature: float,
        cot: bool,
    ) -> CoTResult:
        if self.provider == "simulation":
            return self._simulate(messages, cot)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages, temperature, cot)
        else:
            return self._call_openai(messages, temperature, cot)

    # ── OpenAI backend ───────────────────────────────────────────────────────

    def _call_openai(
        self,
        messages: List[dict],
        temperature: float,
        cot: bool,
    ) -> CoTResult:
        try:
            from openai import OpenAI
        except ImportError:
            return self._simulate(messages, cot)

        client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
        if cot:
            messages = self._inject_cot_prompt(messages)

        resp = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        return self._parse_cot(text, cot)

    # ── Anthropic backend ────────────────────────────────────────────────────

    def _call_anthropic(
        self,
        messages: List[dict],
        temperature: float,
        cot: bool,
    ) -> CoTResult:
        try:
            import anthropic
        except ImportError:
            return self._simulate(messages, cot)

        # Convert OpenAI-format messages to Anthropic format
        system = ""
        anthropic_msgs = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                anthropic_msgs.append({"role": m["role"], "content": m["content"]})

        client = anthropic.Anthropic(api_key=self.api_key)
        kwargs = dict(
            model=self.model,
            max_tokens=4096,
            messages=anthropic_msgs,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system
        if cot:
            kwargs["system"] = (system + "\n\n" if system else "") + (
                "Before giving your final answer, think through this step by step. "
                "Wrap your reasoning in <thinking>...</thinking> tags, "
                "then provide your final answer."
            )

        resp = client.messages.create(**kwargs)
        text = "".join(
            b.text for b in resp.content if hasattr(b, "text")
        )
        return self._parse_cot(text, cot)

    # ── Simulation backend (for demo without API keys) ───────────────────────

    def _simulate(self, messages: List[dict], cot: bool) -> CoTResult:
        """Simulate LLM responses for demonstration purposes."""
        # Extract the last user message as the query
        user_msg = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_msg = m["content"]
                break

        system_msg = ""
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
                break

        reasoning, answer = self._generate_simulation(system_msg, user_msg, cot)
        return CoTResult(reasoning=reasoning, answer=answer, raw_response="", tokens_used=0)

    def _generate_simulation(
        self, system: str, query: str, cot: bool
    ) -> Tuple[str, str]:
        """Generate simulated reasoning based on query content."""
        reasoning = ""
        answer = ""

        if "decompose" in system.lower() or "claim" in system.lower():
            reasoning = self._sim_decompose(query)
            answer = reasoning.split("FINAL_ANSWER:")[-1] if "FINAL_ANSWER:" in reasoning else reasoning
        elif "cross-reference" in system.lower() or "relation" in system.lower():
            reasoning = self._sim_cross_reference(query)
            answer = reasoning.split("FINAL_ANSWER:")[-1] if "FINAL_ANSWER:" in reasoning else reasoning
        elif "gap" in system.lower():
            reasoning = self._sim_gap_analysis(query)
            answer = reasoning.split("FINAL_ANSWER:")[-1] if "FINAL_ANSWER:" in reasoning else reasoning
        elif "synthesize" in system.lower() or "synthesis" in system.lower():
            reasoning = self._sim_synthesis(query)
            answer = reasoning.split("FINAL_ANSWER:")[-1] if "FINAL_ANSWER:" in reasoning else reasoning
        elif "critic" in system.lower() or "review" in system.lower():
            reasoning = self._sim_critic(query)
            answer = reasoning.split("FINAL_ANSWER:")[-1] if "FINAL_ANSWER:" in reasoning else reasoning
        else:
            reasoning = (
                "Step 1: Analyze the query scope and identify key dimensions.\n"
                "Step 2: Consider alternative perspectives and implications.\n"
                "Step 3: Formulate evidence-based conclusions."
            )
            answer = "Analysis complete based on available evidence."

        if not cot:
            return "", answer
        return reasoning, answer

    def _sim_decompose(self, query: str) -> str:
        return (
            "THINKING:\n"
            "Step 1 — Identify paper structure: Parse abstract, introduction, method, results, conclusion.\n"
            "Step 2 — Extract claims from each section with evidence-level tagging.\n"
            "Step 3 — Classify each claim as hypothesis/finding/method/limitation/future_work.\n"
            "Step 4 — Assign confidence based on evidence strength and statistical rigor.\n"
            "Step 5 — Cross-check claims against paper's internal logic for consistency.\n"
            "FINAL_ANSWER:\n"
            f'{{"claims": ['
            f'{{"type": "finding", "content": "The proposed method outperforms baselines by 12.3% on benchmark X", "confidence": "high"}}, '
            f'{{"type": "method", "content": "Uses transformer-based architecture with multi-head attention", "confidence": "high"}}, '
            f'{{"type": "limitation", "content": "Evaluation limited to English-language datasets only", "confidence": "high"}}, '
            f'{{"type": "hypothesis", "content": "Scaling model parameters yields diminishing returns beyond 100B", "confidence": "medium"}}'
            f']}}'
        )

    def _sim_cross_reference(self, query: str) -> str:
        return (
            "THINKING:\n"
            "Step 1 — Build adjacency matrix of all claims across papers.\n"
            "Step 2 — Compute semantic similarity between claims using embedding alignment.\n"
            "Step 3 — Identify supported/contradicted/extended relationships.\n"
            "Step 4 — Trace citation chains to understand idea provenance.\n"
            "Step 5 — Validate transitive relations (if A supports B and B supports C, does A support C?).\n"
            "FINAL_ANSWER:\n"
            '{"relations": ['
            '{"source": "claim_1", "target": "claim_5", "type": "supports", "explanation": "Claim_5 experimental results replicate claim_1 findings"}, '
            '{"source": "claim_3", "target": "claim_7", "type": "contradicts", "explanation": "Claim_7 shows opposite effect under different conditions"}, '
            '{"source": "claim_2", "target": "claim_8", "type": "extends", "explanation": "Claim_8 generalizes claim_2 approach to multi-modal setting"}'
            ']}'
        )

    def _sim_gap_analysis(self, query: str) -> str:
        return (
            "THINKING:\n"
            "Step 1 — Map the research landscape: enumerate covered sub-topics.\n"
            "Step 2 — Identify under-explored areas by density analysis of the knowledge graph.\n"
            "Step 3 — Detect methodological gaps: what methods are missing from the literature.\n"
            "Step 4 — Identify evaluation gaps: missing benchmarks, demographics, or conditions.\n"
            "Step 5 — Prioritize gaps by potential impact and feasibility.\n"
            "FINAL_ANSWER:\n"
            '{"gaps": ['
            '{"description": "No study evaluates performance on low-resource languages (< 1M speakers)", "significance": "high", "related_papers": ["paper_1", "paper_3"]}, '
            '{"description": "Lack of longitudinal studies measuring long-term effects beyond 6 months", "significance": "high", "related_papers": ["paper_2"]}, '
            '{"description": "No comparison between RL-based and supervised fine-tuning approaches", "significance": "medium", "related_papers": ["paper_1", "paper_4"]}'
            ']}'
        )

    def _sim_synthesis(self, query: str) -> str:
        return (
            "THINKING:\n"
            "Step 1 — Aggregate findings across all analyzed papers by theme.\n"
            "Step 2 — Build consensus narrative: where do papers agree?\n"
            "Step 3 — Identify tensions: where do papers disagree and why?\n"
            "Step 4 — Trace evolution of key ideas across the citation graph.\n"
            "Step 5 — Formulate meta-level insights that go beyond individual papers.\n"
            "Step 6 — Generate structured synthesis report with confidence annotations.\n"
            "FINAL_ANSWER:\n"
            '{"synthesis": {'
            '"consensus": "All papers agree that attention mechanisms improve performance on sequence tasks", '
            '"tensions": "Papers disagree on optimal model scale — one finds diminishing returns after 100B params, another shows continued improvement", '
            '"evolution": "The field has moved from RNN-based to Transformer-based architectures (2017-2024)", '
            '"meta_insights": "The primary bottleneck is now data quality rather than model architecture"'
            '}}'
        )

    def _sim_critic(self, query: str) -> str:
        return (
            "THINKING:\n"
            "Step 1 — Review methodology of each paper for internal validity.\n"
            "Step 2 — Check statistical rigor: sample sizes, significance tests, effect sizes.\n"
            "Step 3 — Identify potential confounding factors and alternative explanations.\n"
            "Step 4 — Assess generalizability of findings across contexts.\n"
            "Step 5 — Evaluate strength of evidence for each major claim.\n"
            "FINAL_ANSWER:\n"
            '{"review": {'
            '"methodology_issues": ["Paper_2 uses small sample size (n=30) limiting statistical power", "Paper_3 lacks ablation studies"], '
            '"strengths": ["Paper_1 provides comprehensive benchmark with 5 datasets", "Paper_4 includes human evaluation"], '
            '"overall_assessment": "Moderate confidence in main findings; further replication needed"'
            '}}'
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _inject_cot_prompt(self, messages: List[dict]) -> List[dict]:
        """Inject chain-of-thought prompting into messages."""
        cot_instruction = (
            "Think through this step by step. Break down the problem, consider "
            "alternatives, and show your reasoning before reaching a conclusion. "
            "Format: first write your reasoning, then write 'FINAL_ANSWER:' followed by the answer."
        )
        new_msgs = []
        for m in messages:
            if m["role"] == "system":
                new_msgs.append({
                    "role": "system",
                    "content": m["content"] + "\n\n" + cot_instruction,
                })
            else:
                new_msgs.append(m)
        return new_msgs

    def _parse_cot(self, text: str, cot_enabled: bool) -> CoTResult:
        """Parse reasoning and answer from response text."""
        if not cot_enabled:
            return CoTResult(reasoning="", answer=text.strip(), raw_response=text)

        if "FINAL_ANSWER:" in text:
            parts = text.split("FINAL_ANSWER:", 1)
            reasoning = parts[0].strip()
            answer = parts[1].strip()
        elif "<thinking>" in text and "</thinking>" in text:
            thinking_match = re.search(r"<thinking>(.*?)</thinking>", text, re.DOTALL)
            reasoning = thinking_match.group(1).strip() if thinking_match else ""
            answer = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()
        else:
            reasoning = text[: len(text) // 2]
            answer = text

        return CoTResult(
            reasoning=reasoning,
            answer=answer,
            raw_response=text,
        )
