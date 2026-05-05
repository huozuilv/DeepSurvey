"""
Multi-agent orchestrator with long-chain reasoning.

Coordinates the five specialized agents through a structured workflow:
  1. Decomposer → extracts claims from papers
  2. CrossReferencer → identifies relations between claims
  3. GapAnalyzer → detects research gaps
  4. Synthesizer → produces coherent synthesis
  5. Critic → reviews the analysis

Features long-chain reasoning: each agent's output feeds into the next,
creating a chain of reasoning that traces from raw papers to final synthesis.
The orchestrator can also perform iterative refinement, sending critique
back to earlier agents for improved results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from .agents import (
    CriticAgent,
    CrossReferenceAgent,
    DecomposerAgent,
    GapAnalyzerAgent,
    SynthesizerAgent,
)
from .knowledge_graph import KnowledgeGraph
from .llm_client import LLMClient
from .models import (
    AgentMessage,
    AnalysisReport,
    Claim,
    Contradiction,
    Paper,
    ReasoningChain,
    ReasoningStep,
    Relation,
    ResearchGap,
)


class WorkflowPhase(str, Enum):
    INIT = "init"
    DECOMPOSE = "decompose"
    CROSS_REFERENCE = "cross_reference"
    GAP_ANALYSIS = "gap_analysis"
    SYNTHESIZE = "synthesize"
    CRITIQUE = "critique"
    REFINE = "refine"
    COMPLETE = "complete"


@dataclass
class WorkflowState:
    """Tracks the state of the multi-agent workflow."""
    phase: WorkflowPhase = WorkflowPhase.INIT
    papers: List[Paper] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    gaps: List[ResearchGap] = field(default_factory=list)
    contradictions: List[Contradiction] = field(default_factory=list)
    synthesis: dict = field(default_factory=dict)
    critique: dict = field(default_factory=dict)
    reasoning_chains: List[ReasoningChain] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)
    iteration: int = 0
    started_at: datetime = field(default_factory=datetime.now)


class ResearchOrchestrator:
    """
    Coordinates multi-agent research synthesis with long-chain reasoning.

    The orchestrator manages a pipeline of specialized agents, passing
    outputs from one to the next, building up a comprehensive analysis.
    Supports iterative refinement where the Critic's feedback triggers
    re-analysis by earlier agents.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        max_iterations: int = 3,
        verbose: bool = True,
    ):
        self.kg = KnowledgeGraph()
        self.llm = llm or LLMClient()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.state = WorkflowState()

        # Initialize agents with shared knowledge graph
        self.decomposer = DecomposerAgent(self.kg, self.llm, verbose=verbose)
        self.cross_referencer = CrossReferenceAgent(self.kg, self.llm, verbose=verbose)
        self.gap_analyzer = GapAnalyzerAgent(self.kg, self.llm, verbose=verbose)
        self.synthesizer = SynthesizerAgent(self.kg, self.llm, verbose=verbose)
        self.critic = CriticAgent(self.kg, self.llm, verbose=verbose)

    # ── Main Pipeline ────────────────────────────────────────────────────────

    def analyze(self, papers: List[Paper]) -> AnalysisReport:
        """
        Run the full multi-agent analysis pipeline on a set of papers.

        Pipeline: Decompose → Cross-Reference → Gap Analysis → Synthesize → Critique
        With optional iterative refinement loop.
        """
        self.state = WorkflowState(papers=papers)

        self._log_header("DeepSurvey Analysis Pipeline Started")
        self._log(f"Papers: {len(papers)}, Max iterations: {self.max_iterations}")

        # Phase 1: Load papers into knowledge graph
        self._phase_load_papers(papers)

        # Phase 2: Decompose papers into claims
        self._phase_decompose(papers)

        # Phase 3: Cross-reference claims across papers
        self._phase_cross_reference()

        # Phase 4: Gap analysis
        self._phase_gap_analysis()

        # Phase 5: Synthesize findings
        self._phase_synthesize()

        # Phase 6: Critical review
        self._phase_critique()

        # Phase 7: Iterative refinement (if needed)
        while self.state.iteration < self.max_iterations:
            if self._should_refine():
                self.state.iteration += 1
                self._log(f"\n--- Refinement Iteration {self.state.iteration} ---")
                self._phase_refine()
            else:
                break

        self.state.phase = WorkflowPhase.COMPLETE
        self._log_header("Analysis Complete")

        return self._build_report()

    # ── Pipeline Phases ──────────────────────────────────────────────────────

    def _phase_load_papers(self, papers: List[Paper]) -> None:
        self._log_header("Phase 1: Loading Papers into Knowledge Graph")
        for paper in papers:
            self.kg.add_paper(paper)
            self._log(f"  Loaded: {paper.title[:60]}...")
        self._log(f"  Total: {self.kg.paper_count} papers in knowledge graph")

    def _phase_decompose(self, papers: List[Paper]) -> None:
        self._log_header("Phase 2: Paper Decomposition (Long-Chain Reasoning)")
        self.state.phase = WorkflowPhase.DECOMPOSE

        all_claims = []
        for i, paper in enumerate(papers):
            self._log(f"  [{i+1}/{len(papers)}] Decomposing: {paper.title[:50]}...")
            claims = self.decomposer.decompose_paper(paper)
            for claim in claims:
                self.kg.add_claim(claim)
                all_claims.append(claim)
            self._log(f"      Extracted {len(claims)} claims")

        self.state.claims = all_claims
        self.state.reasoning_chains.extend(self.decomposer.reasoning_traces)
        self._log(f"  Total claims extracted: {len(all_claims)}")

    def _phase_cross_reference(self) -> None:
        self._log_header("Phase 3: Cross-Reference Analysis")
        self.state.phase = WorkflowPhase.CROSS_REFERENCE

        relations = self.cross_referencer.cross_reference(
            self.state.claims, self.state.papers
        )
        for rel in relations:
            self.kg.add_relation(rel)

        self.state.relations = relations
        self.state.reasoning_chains.extend(self.cross_referencer.reasoning_traces)
        self._log(f"  Identified {len(relations)} cross-claim relations")

    def _phase_gap_analysis(self) -> None:
        self._log_header("Phase 4: Research Gap Analysis")
        self.state.phase = WorkflowPhase.GAP_ANALYSIS

        gaps = self.gap_analyzer.analyze_gaps(
            self.state.papers, self.state.claims, self.state.relations
        )
        self.state.gaps = gaps
        self.state.reasoning_chains.extend(self.gap_analyzer.reasoning_traces)
        self._log(f"  Identified {len(gaps)} research gaps")
        for g in gaps[:3]:
            self._log(f"    - {g.description[:80]}... (significance: {g.significance})")

    def _phase_synthesize(self) -> None:
        self._log_header("Phase 5: Multi-Paper Synthesis")
        self.state.phase = WorkflowPhase.SYNTHESIZE

        # Detect contradictions from the knowledge graph
        self.state.contradictions = self.kg.detect_contradictions()

        synthesis = self.synthesizer.synthesize(
            self.state.papers,
            self.state.claims,
            self.state.relations,
            self.state.gaps,
            self.state.contradictions,
        )
        self.state.synthesis = synthesis
        self.state.reasoning_chains.extend(self.synthesizer.reasoning_traces)

        self._log(f"  Consensus: {synthesis.get('consensus', '')[:80]}...")
        self._log(f"  Meta-insights: {synthesis.get('meta_insights', '')[:80]}...")

    def _phase_critique(self) -> None:
        self._log_header("Phase 6: Critical Review")
        self.state.phase = WorkflowPhase.CRITIQUE

        critique = self.critic.review(
            self.state.papers, self.state.claims, self.state.synthesis
        )
        self.state.critique = critique
        self.state.reasoning_chains.extend(self.critic.reasoning_traces)

        issues = critique.get("methodology_issues", [])
        self._log(f"  Methodology issues found: {len(issues)}")
        for issue in issues[:3]:
            self._log(f"    - {issue[:80]}")
        self._log(f"  Overall: {critique.get('overall_assessment', '')[:80]}")

    def _should_refine(self) -> bool:
        """Determine if refinement is needed based on critique."""
        issues = self.state.critique.get("methodology_issues", [])
        overstated = self.state.critique.get("overstated_claims", [])
        return (len(issues) + len(overstated)) > 2 and self.state.iteration < self.max_iterations

    def _phase_refine(self) -> None:
        """Refine synthesis based on critique feedback."""
        self.state.phase = WorkflowPhase.REFINE

        # Feed critique back to synthesizer for refinement
        critique_feedback = json.dumps(self.state.critique, indent=2)
        self._log("  Feeding critique back for refinement...")

        # Re-synthesize with critique as additional context
        refined = self.synthesizer.synthesize(
            self.state.papers,
            self.state.claims,
            self.state.relations,
            self.state.gaps,
            self.state.contradictions,
        )
        self.state.synthesis = refined
        self.state.reasoning_chains.extend(self.synthesizer.reasoning_traces)

    # ── Report Generation ────────────────────────────────────────────────────

    def _build_report(self) -> AnalysisReport:
        """Build the final analysis report from workflow state."""
        stats = self.kg.get_statistics()

        # Compile key findings from claims
        findings = [
            c.content
            for c in self.state.claims
            if c.type.value == "finding"
        ]

        # Compile future directions
        future = list(self.state.synthesis.get("future_directions", []))
        for g in self.state.gaps:
            if g.suggested_approach:
                future.append(f"[Gap] {g.description[:80]} → {g.suggested_approach[:80]}")

        # Paper list for report
        paper_list = [p.title for p in self.state.papers]

        return AnalysisReport(
            title=f"Multi-Agent Research Synthesis: {len(self.state.papers)} Papers Analyzed",
            papers_analyzed=paper_list,
            executive_summary=self.state.synthesis.get("consensus", ""),
            key_findings=findings[:10],
            contradictions=self.state.contradictions,
            research_gaps=self.state.gaps,
            methodology_comparison=self.state.critique.get("overall_assessment", ""),
            future_directions=future[:10],
            reasoning_traces=self.state.reasoning_chains,
            knowledge_graph_stats=stats,
        )

    # ── Interactive Query ────────────────────────────────────────────────────

    def query(self, question: str) -> str:
        """
        Answer a question using the accumulated knowledge.
        Uses long-chain reasoning across the knowledge graph.
        """
        chain = ReasoningChain(question=question)

        # Step 1: Identify relevant claims
        relevant_claims = []
        question_lower = question.lower()
        for claim in self.state.claims:
            if any(word in claim.content.lower() for word in question_lower.split()):
                relevant_claims.append(claim)

        chain.steps.append(
            ReasoningStep(
                agent="Orchestrator",
                step_number=1,
                action="retrieve",
                thought=f"Searching for claims relevant to: '{question}'",
                observation=f"Found {len(relevant_claims)} potentially relevant claims",
                timestamp=datetime.now(),
            )
        )

        # Step 2: Trace relations from relevant claims
        related_nodes = set()
        for claim in relevant_claims[:10]:
            neighborhood = self.kg.neighborhood_summary(claim.id)
            related_nodes.add(claim.id)

        chain.steps.append(
            ReasoningStep(
                agent="Orchestrator",
                step_number=2,
                action="graph_traverse",
                thought="Tracing relations from relevant claims through knowledge graph",
                observation=f"Expanded to {len(related_nodes)} related nodes",
                timestamp=datetime.now(),
            )
        )

        # Step 3: Synthesize answer
        context = "\n".join(
            f"  [{c.type.value}] {c.content[:120]}" for c in relevant_claims[:10]
        )

        system = (
            "You are a research synthesis assistant. Use the provided claims and "
            "relations to answer the question. Cite specific claims when possible. "
            "Think step by step before answering."
        )

        answer = self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": f"Claims:\n{context}\n\nQuestion: {question}"},
        ])

        chain.steps.append(
            ReasoningStep(
                agent="Orchestrator",
                step_number=3,
                action="synthesize",
                thought="Synthesizing answer from retrieved claims",
                conclusion=answer[:200],
                timestamp=datetime.now(),
            )
        )
        chain.final_answer = answer

        self.state.reasoning_chains.append(chain)
        return answer

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _log_header(self, title: str) -> None:
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  {title}")
            print(f"{'='*60}")
