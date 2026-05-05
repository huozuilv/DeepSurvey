"""
Core data models for DeepSurvey — Deep Multi-Agent Literature Synthesis.
Defines the ontology of academic entities used across all agents.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Identifiers ──────────────────────────────────────────────────────────────

def _uid() -> str:
    return uuid.uuid4().hex[:12]


# ── Enums ────────────────────────────────────────────────────────────────────

class ClaimType(str, Enum):
    HYPOTHESIS = "hypothesis"
    FINDING = "finding"
    METHOD = "method"
    LIMITATION = "limitation"
    FUTURE_WORK = "future_work"
    DEFINITION = "definition"
    ASSUMPTION = "assumption"


class RelationType(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    CITES = "cites"
    USES_METHOD = "uses_method"
    ADDRESSES_LIMITATION = "addresses_limitation"
    MOTIVATES = "motivates"
    EQUIVALENT_TO = "equivalent_to"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ── Core entities ────────────────────────────────────────────────────────────

class Author(BaseModel):
    name: str
    affiliation: Optional[str] = None


class Paper(BaseModel):
    """A research paper under analysis."""
    id: str = Field(default_factory=_uid)
    title: str
    authors: List[Author] = Field(default_factory=list)
    year: int = 2024
    abstract: str = ""
    venue: str = ""
    sections: Dict[str, str] = Field(default_factory=dict)
    citations: List[str] = Field(default_factory=list)
    full_text: str = ""


class Claim(BaseModel):
    """A structured claim extracted from a paper."""
    id: str = Field(default_factory=_uid)
    paper_id: str
    type: ClaimType
    content: str
    evidence: str = ""
    confidence: Confidence = Confidence.MEDIUM
    section: str = ""
    related_claims: List[str] = Field(default_factory=list)


class Relation(BaseModel):
    """A directed relationship between two claims or papers."""
    id: str = Field(default_factory=_uid)
    source_id: str
    target_id: str
    type: RelationType
    explanation: str = ""
    confidence: Confidence = Confidence.MEDIUM


class ResearchGap(BaseModel):
    """An identified gap in the research landscape."""
    id: str = Field(default_factory=_uid)
    description: str
    related_claims: List[str] = Field(default_factory=list)
    related_papers: List[str] = Field(default_factory=list)
    significance: str = ""
    suggested_approach: str = ""


class Contradiction(BaseModel):
    """A contradiction between two claims."""
    id: str = Field(default_factory=_uid)
    claim_a_id: str
    claim_b_id: str
    description: str
    resolution: str = ""
    severity: str = "unresolved"


# ── Reasoning ────────────────────────────────────────────────────────────────

class ReasoningStep(BaseModel):
    """A single step in a long-chain reasoning trace."""
    id: str = Field(default_factory=_uid)
    agent: str
    step_number: int
    action: str
    thought: str
    observation: str = ""
    conclusion: str = ""
    references: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)


class ReasoningChain(BaseModel):
    """A complete chain-of-thought reasoning trace."""
    id: str = Field(default_factory=_uid)
    question: str
    steps: List[ReasoningStep] = Field(default_factory=list)
    final_answer: str = ""
    confidence: Confidence = Confidence.MEDIUM

    @property
    def step_count(self) -> int:
        return len(self.steps)


# ── Agent messages ───────────────────────────────────────────────────────────

class AgentMessage(BaseModel):
    """Message passed between agents."""
    id: str = Field(default_factory=_uid)
    sender: str
    receiver: str
    type: str  # "request", "response", "broadcast", "query"
    content: str
    data: dict = Field(default_factory=dict)
    in_reply_to: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class AnalysisReport(BaseModel):
    """Final synthesis report produced by the multi-agent system."""
    title: str
    papers_analyzed: List[str] = Field(default_factory=list)
    executive_summary: str = ""
    key_findings: List[str] = Field(default_factory=list)
    contradictions: List[Contradiction] = Field(default_factory=list)
    research_gaps: List[ResearchGap] = Field(default_factory=list)
    methodology_comparison: str = ""
    future_directions: List[str] = Field(default_factory=list)
    reasoning_traces: List[ReasoningChain] = Field(default_factory=list)
    knowledge_graph_stats: dict = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=datetime.now)
