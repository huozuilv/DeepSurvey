"""
Specialized agents for DeepSurvey — Deep Multi-Agent Literature Synthesis.

Agents:
  - DecomposerAgent: Extracts structured claims from papers
  - CrossReferenceAgent: Identifies relations between claims across papers
  - GapAnalyzerAgent: Identifies research gaps
  - SynthesizerAgent: Synthesizes findings into coherent narratives
  - CriticAgent: Critical review of methodology and evidence quality
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from .knowledge_graph import KnowledgeGraph
from .llm_client import CoTResult, LLMClient
from .models import (
    AgentMessage,
    Claim,
    ClaimType,
    Confidence,
    Contradiction,
    Paper,
    ReasoningChain,
    ReasoningStep,
    Relation,
    RelationType,
    ResearchGap,
)


# ── Base Agent ───────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base for all specialized agents."""

    name: str = "base"
    role: str = "base"

    def __init__(
        self,
        kg: KnowledgeGraph,
        llm: LLMClient,
        verbose: bool = False,
    ):
        self.kg = kg
        self.llm = llm
        self.verbose = verbose
        self.inbox: list[AgentMessage] = []
        self.reasoning_traces: list[ReasoningChain] = []

    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt defining this agent's role and behavior."""

    def receive(self, msg: AgentMessage) -> None:
        """Receive a message from another agent."""
        self.inbox.append(msg)
        if self.verbose:
            print(f"  [{self.name}] ← {msg.sender}: {msg.content[:80]}...")

    def send(self, receiver: str, msg_type: str, content: str, data: dict | None = None) -> AgentMessage:
        """Create a message to send to another agent."""
        return AgentMessage(
            sender=self.name,
            receiver=receiver,
            type=msg_type,
            content=content,
            data=data or {},
        )

    def think(self, question: str, context: str = "") -> ReasoningChain:
        """
        Perform chain-of-thought reasoning on a question.
        Records the full reasoning trace.
        """
        system = self.system_prompt()
        if context:
            system += f"\n\nContext:\n{context}"

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        result: CoTResult = self.llm.chain_of_thought(messages)

        chain = ReasoningChain(
            question=question,
            final_answer=result.answer,
            confidence=Confidence.MEDIUM,
        )

        # Parse reasoning into steps
        if result.reasoning:
            steps_text = result.reasoning.split("\n")
            step_num = 0
            current_step_lines = []
            for line in steps_text:
                if re.match(r"^Step\s+\d+", line) and current_step_lines:
                    step_num += 1
                    chain.steps.append(
                        ReasoningStep(
                            agent=self.name,
                            step_number=step_num,
                            action="reasoning",
                            thought="\n".join(current_step_lines),
                            timestamp=datetime.now(),
                        )
                    )
                    current_step_lines = [line]
                else:
                    current_step_lines.append(line)
            if current_step_lines:
                step_num += 1
                chain.steps.append(
                    ReasoningStep(
                        agent=self.name,
                        step_number=step_num,
                        action="reasoning",
                        thought="\n".join(current_step_lines),
                        timestamp=datetime.now(),
                    )
                )

        self.reasoning_traces.append(chain)
        return chain

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [{self.name}] {msg}")


# ── Decomposer Agent ─────────────────────────────────────────────────────────

class DecomposerAgent(BaseAgent):
    """
    Decomposes research papers into structured claims.
    Extracts hypotheses, findings, methods, limitations, and future work.
    """

    name = "Decomposer"
    role = "Paper Decomposition Specialist"

    def system_prompt(self) -> str:
        return (
            "You are an expert in decomposing academic papers into structured claims. "
            "Your task is to read a paper and extract: "
            "1) Hypotheses — what the paper predicts or assumes, "
            "2) Findings — empirical results and conclusions, "
            "3) Methods — techniques and approaches used, "
            "4) Limitations — acknowledged weaknesses and constraints, "
            "5) Future work — suggested next steps. "
            "For each claim, assess confidence (high/medium/low) based on evidence strength. "
            "Output claims as a JSON array."
        )

    def decompose_paper(self, paper: Paper) -> list[Claim]:
        """Decompose a single paper into structured claims."""
        self._log(f"Decomposing: {paper.title[:60]}...")

        query = (
            f"Paper: {paper.title}\n"
            f"Authors: {', '.join(a.name for a in paper.authors)}\n"
            f"Year: {paper.year}\n"
            f"Venue: {paper.venue}\n"
            f"Abstract: {paper.abstract}\n"
        )
        if paper.full_text:
            # Include section summaries if full text available
            query += f"\nFull Text Sections: {list(paper.sections.keys())}\n"
            query += paper.full_text[:4000]

        query += "\nExtract all claims from this paper as structured JSON."

        chain = self.think(query, f"Paper ID: {paper.id}")
        claims = self._parse_claims(chain.final_answer, paper.id)
        return claims

    def _parse_claims(self, raw: str, paper_id: str) -> list[Claim]:
        """Parse claims from LLM output, with fallback extraction."""
        claims = []

        # Try JSON parsing
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                claim_list = data.get("claims", [data])
                for item in claim_list:
                    claims.append(
                        Claim(
                            paper_id=paper_id,
                            type=ClaimType(item.get("type", "finding")),
                            content=item.get("content", str(item)[:200]),
                            evidence=item.get("evidence", ""),
                            confidence=Confidence(item.get("confidence", "medium")),
                        )
                    )
                return claims
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: parse structured text
        for line in raw.split("\n"):
            line = line.strip()
            if not line or len(line) < 10:
                continue
            for prefix, ctype in [
                ("Hypothesis:", ClaimType.HYPOTHESIS),
                ("Finding:", ClaimType.FINDING),
                ("Method:", ClaimType.METHOD),
                ("Limitation:", ClaimType.LIMITATION),
                ("Future Work:", ClaimType.FUTURE_WORK),
            ]:
                if prefix in line:
                    content = line.split(prefix, 1)[1].strip()
                    claims.append(
                        Claim(
                            paper_id=paper_id,
                            type=ctype,
                            content=content[:300],
                        )
                    )
                    break

        return claims


# ── Cross-Reference Agent ────────────────────────────────────────────────────

class CrossReferenceAgent(BaseAgent):
    """
    Identifies relationships between claims across papers.
    Detects support, contradiction, extension, and methodological relationships.
    """

    name = "CrossReferencer"
    role = "Cross-Reference Analysis Specialist"

    def system_prompt(self) -> str:
        return (
            "You are an expert in identifying relationships between research claims "
            "across different papers. You can detect: "
            "1) SUPPORTS — one finding corroborates another, "
            "2) CONTRADICTS — findings that disagree, "
            "3) EXTENDS — one work generalizes or builds on another, "
            "4) CITES — direct citation relationships, "
            "5) USES_METHOD — methodological borrowing. "
            "For each relation, explain the nature of the connection and your confidence."
        )

    def cross_reference(
        self,
        claims: list[Claim],
        papers: list[Paper],
    ) -> list[Relation]:
        """Find all cross-claim relations across papers."""
        self._log(f"Cross-referencing {len(claims)} claims from {len(papers)} papers...")

        relations: list[Relation] = []

        # Build claim context
        claim_descriptions = []
        for c in claims:
            paper = self.kg.get_paper(c.paper_id)
            paper_title = paper.title[:50] if paper else "Unknown"
            claim_descriptions.append(
                f"  [{c.id}] ({c.type.value}) from '{paper_title}': {c.content[:120]}"
            )

        query = (
            "IDENTIFY RELATIONS between the following claims. "
            "For each pair that has a meaningful relationship, specify:\n"
            "- source claim ID\n"
            "- target claim ID\n"
            "- relation type (supports/contradicts/extends/cites/uses_method)\n"
            "- explanation of the relationship\n\n"
            "CLAIMS:\n" + "\n".join(claim_descriptions[:20])
        )

        chain = self.think(query)
        parsed = self._parse_relations(chain.final_answer)
        relations.extend(parsed)

        # Also add explicit citation relations from paper metadata
        for paper in papers:
            for cited_id in paper.citations:
                if cited_id in self.kg._papers:
                    relations.append(
                        Relation(
                            source_id=paper.id,
                            target_id=cited_id,
                            type=RelationType.CITES,
                            explanation=f"{paper.title[:40]} cites this work",
                        )
                    )

        return relations

    def _parse_relations(self, raw: str) -> list[Relation]:
        """Parse relations from LLM output."""
        relations = []
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                rel_list = data.get("relations", [data])
                for item in rel_list:
                    relations.append(
                        Relation(
                            source_id=item.get("source", item.get("source_id", "")),
                            target_id=item.get("target", item.get("target_id", "")),
                            type=RelationType(item.get("type", "cites")),
                            explanation=item.get("explanation", ""),
                            confidence=Confidence(item.get("confidence", "medium")),
                        )
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return relations


# ── Gap Analyzer Agent ───────────────────────────────────────────────────────

class GapAnalyzerAgent(BaseAgent):
    """
    Identifies research gaps by analyzing the knowledge graph structure
    and combining it with LLM-based semantic gap detection.
    """

    name = "GapAnalyzer"
    role = "Research Gap Identification Specialist"

    def system_prompt(self) -> str:
        return (
            "You are an expert in identifying research gaps and unexplored areas "
            "in academic literature. Look for: "
            "1) Underexplored topics with sparse coverage, "
            "2) Methodological gaps — missing approaches or comparisons, "
            "3) Evaluation gaps — missing benchmarks, populations, or conditions, "
            "4) Theoretical gaps — missing formal frameworks, "
            "5) Application gaps — findings not tested in real-world settings. "
            "For each gap, assess significance and suggest possible approaches."
        )

    def analyze_gaps(
        self,
        papers: list[Paper],
        claims: list[Claim],
        relations: list[Relation],
    ) -> list[ResearchGap]:
        """Identify research gaps from the literature landscape."""
        self._log("Analyzing research gaps...")

        gaps: list[ResearchGap] = []

        # 1. Structural gaps from graph analysis
        structural = self.kg.identify_structural_gaps()
        gaps.extend(structural)
        self._log(f"  Found {len(structural)} structural gaps")

        # 2. LLM-based semantic gap analysis
        claim_summary = "\n".join(
            f"  [{c.type.value}] {c.content[:100]}" for c in claims[:15]
        )
        paper_summary = "\n".join(
            f"  - {p.title[:80]} ({p.year})" for p in papers
        )

        query = (
            "IDENTIFY RESEARCH GAPS based on the following literature landscape:\n\n"
            f"PAPERS ({len(papers)}):\n{paper_summary}\n\n"
            f"CLAIMS:\n{claim_summary}\n\n"
            "Identify 3-5 specific research gaps. For each, provide:\n"
            "- description of the gap\n"
            "- significance (high/medium/low)\n"
            "- suggested approach to address it\n"
            "- which papers are related"
        )

        chain = self.think(query, f"Analyzing {len(papers)} papers with {len(claims)} claims")
        semantic_gaps = self._parse_gaps(chain.final_answer)
        gaps.extend(semantic_gaps)
        self._log(f"  Found {len(semantic_gaps)} semantic gaps")

        return gaps

    def _parse_gaps(self, raw: str) -> list[ResearchGap]:
        """Parse research gaps from LLM output."""
        gaps = []
        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                gap_list = data.get("gaps", [data])
                for item in gap_list:
                    gaps.append(
                        ResearchGap(
                            description=item.get("description", str(item)[:200]),
                            significance=item.get("significance", "medium"),
                            suggested_approach=item.get("suggested_approach", ""),
                            related_papers=item.get("related_papers", []),
                        )
                    )
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return gaps


# ── Synthesizer Agent ────────────────────────────────────────────────────────

class SynthesizerAgent(BaseAgent):
    """
    Synthesizes findings across papers into coherent narratives.
    Produces consensus statements, identifies tensions, and traces idea evolution.
    """

    name = "Synthesizer"
    role = "Research Synthesis Specialist"

    def system_prompt(self) -> str:
        return (
            "You are an expert in synthesizing research findings across multiple papers. "
            "Your synthesis should: "
            "1) Identify areas of consensus across papers, "
            "2) Highlight tensions and disagreements, "
            "3) Trace the evolution of key ideas through the literature, "
            "4) Generate meta-level insights beyond individual papers, "
            "5) Provide a structured narrative with confidence assessments. "
            "Think carefully about each dimension before reaching conclusions."
        )

    def synthesize(
        self,
        papers: list[Paper],
        claims: list[Claim],
        relations: list[Relation],
        gaps: list[ResearchGap],
        contradictions: list[Contradiction],
    ) -> dict:
        """
        Produce a comprehensive synthesis of the literature.
        Returns structured synthesis data.
        """
        self._log(f"Synthesizing {len(papers)} papers...")

        # Build context
        findings = [c for c in claims if c.type == ClaimType.FINDING]
        methods = [c for c in claims if c.type == ClaimType.METHOD]

        context_parts = [
            f"Papers analyzed: {len(papers)}",
            f"Total claims extracted: {len(claims)}",
            f"Findings: {len(findings)}, Methods: {len(methods)}",
            f"Cross-relations identified: {len(relations)}",
            f"Contradictions detected: {len(contradictions)}",
            f"Research gaps identified: {len(gaps)}",
        ]

        for c in contradictions[:5]:
            context_parts.append(
                f"Contradiction: {c.description[:100]}"
            )

        for g in gaps[:5]:
            context_parts.append(f"Gap: {g.description[:100]}")

        context = "\n".join(context_parts)

        query = (
            "SYNTHESIZE the following research landscape. Produce:\n"
            "1) Consensus narrative — what do these papers agree on?\n"
            "2) Tensions — what do they disagree on and why?\n"
            "3) Idea evolution — how have key concepts evolved?\n"
            "4) Meta-insights — what patterns emerge across all papers?\n"
            "5) Future directions — where should the field go next?\n\n"
            "Think through each dimension carefully before writing."
        )

        chain = self.think(query, context)
        return self._parse_synthesis(chain.final_answer, chain)

    def _parse_synthesis(self, raw: str, chain: ReasoningChain) -> dict:
        """Parse synthesis output."""
        result = {
            "consensus": "",
            "tensions": "",
            "evolution": "",
            "meta_insights": "",
            "future_directions": [],
        }

        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                synth = data.get("synthesis", data)
                result.update({
                    "consensus": synth.get("consensus", ""),
                    "tensions": synth.get("tensions", ""),
                    "evolution": synth.get("evolution", ""),
                    "meta_insights": synth.get("meta_insights", ""),
                    "future_directions": synth.get("future_directions", []),
                })
                return result
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback: extract by sections
        sections = {
            "CONSENSUS": "consensus",
            "TENSIONS": "tensions",
            "EVOLUTION": "evolution",
            "META": "meta_insights",
            "FUTURE": "future_directions",
        }
        current_section = "consensus"
        buffers: dict[str, list[str]] = {k: [] for k in sections.values()}

        for line in raw.split("\n"):
            line_stripped = line.strip()
            for keyword, key in sections.items():
                if keyword in line_stripped.upper() and len(line_stripped) < 40:
                    current_section = key
                    break
            else:
                buffers[current_section].append(line)

        for key, lines in buffers.items():
            val = "\n".join(lines).strip()
            if key == "future_directions":
                result[key] = [l.strip("- ") for l in lines if l.strip()]
            else:
                result[key] = val

        return result


# ── Critic Agent ─────────────────────────────────────────────────────────────

class CriticAgent(BaseAgent):
    """
    Critical review agent that evaluates methodology, evidence quality,
    and internal validity of papers and claims.
    """

    name = "Critic"
    role = "Critical Review Specialist"

    def system_prompt(self) -> str:
        return (
            "You are a rigorous academic reviewer. Your task is to critically evaluate "
            "research papers and claims for: "
            "1) Methodological soundness — are methods appropriate and correctly applied? "
            "2) Statistical rigor — sample sizes, significance, effect sizes, "
            "3) Internal validity — are conclusions supported by the evidence? "
            "4) Generalizability — do findings extend beyond the study context? "
            "5) Alternative explanations — are there unaddressed confounding factors? "
            "Be constructively critical but fair."
        )

    def review(
        self,
        papers: list[Paper],
        claims: list[Claim],
        synthesis: dict,
    ) -> dict:
        """Perform critical review of the entire analysis."""
        self._log(f"Critically reviewing {len(papers)} papers...")

        key_claims = [c for c in claims if c.confidence in (Confidence.HIGH, Confidence.MEDIUM)]
        claim_texts = "\n".join(
            f"  [{c.type.value}] {c.content[:100]} (confidence: {c.confidence.value})"
            for c in key_claims[:15]
        )

        query = (
            "CRITICALLY REVIEW the following research findings:\n\n"
            f"CLAIMS TO REVIEW:\n{claim_texts}\n\n"
            f"SYNTHESIS DRAFT:\n{synthesis.get('consensus', '')[:500]}\n\n"
            "Evaluate:\n"
            "1) Methodology issues across the papers\n"
            "2) Strengths of the overall evidence base\n"
            "3) Claims that may be overstated\n"
            "4) Overall confidence assessment"
        )

        chain = self.think(query)
        return self._parse_review(chain.final_answer)

    def _parse_review(self, raw: str) -> dict:
        """Parse critic review."""
        result = {
            "methodology_issues": [],
            "strengths": [],
            "overstated_claims": [],
            "overall_assessment": "",
        }

        try:
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                review = data.get("review", data)
                result.update({
                    "methodology_issues": review.get("methodology_issues", []),
                    "strengths": review.get("strengths", []),
                    "overstated_claims": review.get("overstated_claims", []),
                    "overall_assessment": review.get("overall_assessment", ""),
                })
                return result
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        # Fallback
        result["overall_assessment"] = raw[:500]
        return result
