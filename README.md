# DeepSurvey — Deep Multi-Agent Literature Synthesis

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<p align="center">
  <em><strong>From paper pile to insight — five specialized agents, one continuous chain of reasoning.</strong></em>
</p>

---

## What is DeepSurvey?

DeepSurvey is an **agent-driven academic literature synthesis engine**. Given a collection of research papers, it deploys five specialized AI agents that collaborate through long-chain reasoning to:

1. **Decompose** each paper into structured claims
2. **Cross-reference** those claims into a knowledge graph of relations
3. **Detect** research gaps via structural and semantic analysis
4. **Synthesize** a narrative report covering consensus, tensions, and evolution
5. **Critique** the evidence — methodology, statistical rigor, generalizability

The output is a **comprehensive synthesis report** — not a keyword summary, but a traceable, evidence-anchored analysis backed by an explicit reasoning chain you can audit.

### Why "DeepSurvey"?

| Concept | Meaning in DeepSurvey |
|---------|----------------------|
| **Deep** | Long-chain reasoning across multiple hops in the knowledge graph; iterative critique-refine loops; claims traced to their source papers |
| **Survey** | The traditional academic literature survey, automated — systematic coverage, gap identification, cross-paper synthesis |

---

## The Core Pain Point

Academic literature review is **broken by scale**:

| Pain Point | Status Quo | DeepSurvey |
|-------------|-----------|------------|
| **Volume** — 8,000+ papers/day on arXiv | Researchers sample heuristically | 5 agents process all inputs systematically |
| **Fragmentation** — findings scattered across venues | Manual cross-referencing | Knowledge graph with 7 relation types |
| **Blind spots** — gaps invisible without global view | Discovered accidentally, years later | Structural + semantic gap detection |
| **Synthesis cost** — weeks to months per survey | One researcher, one brain | Five specialized agents, continuous reasoning |
| **Traceability** — notes lose provenance | "Where did I read that?" | Every claim linked to its paper; every reasoning step recorded |

---

## Architecture

```
User Input: 5 papers on AI Alignment
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│                   Orchestrator                          │
│            (long-chain reasoning engine)                │
│                                                         │
│   Phase 1           Phase 2           Phase 3           │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐       │
│   │Decomposer│────▶│  Cross-  │────▶│   Gap    │       │
│   │ Paper →  │     │Reference │     │ Analyzer │       │
│   │ Claims   │     │ Claims → │     │ Graph +  │       │
│   └──────────┘     │Relations │     │ Semantic │       │
│                    └──────────┘     └──────────┘       │
│                         │                 │             │
│                         ▼                 ▼             │
│                    ┌──────────────────────────┐        │
│                    │     Knowledge Graph       │        │
│                    │   (networkx.DiGraph)      │        │
│                    │                           │        │
│                    │  Paper ──▶ Claim ──▶ Claim│        │
│                    │           supports/       │        │
│                    │         contradicts/      │        │
│                    │           extends          │        │
│                    └──────────────────────────┘        │
│                         │                 │             │
│   Phase 4               ▼        Phase 5  ▼             │
│   ┌──────────┐     ┌──────────────────────────┐        │
│   │Synthesizer│◀───│        Critic            │        │
│   │ Claims +  │───▶│   Methodology Review     │        │
│   │Relations→ │    │   Evidence Assessment    │        │
│   │Narrative  │    └──────────────────────────┘        │
│   └──────────┘              │                          │
│        │                    │ (refine if needed)       │
│        └────────────────────┘                          │
│                    │                                   │
└────────────────────┼───────────────────────────────────┘
                     ▼
            ┌─────────────────┐
            │ Synthesis Report │
            │  • Executive     │
            │    Summary       │
            │  • Key Findings  │
            │  • Contradictions│
            │  • Research Gaps │
            │  • Future        │
            │    Directions    │
            │  • Reasoning     │
            │    Traces        │
            └─────────────────┘
```

### Five Agents, One Mission

| Agent | Role | Output | Key Capability |
|-------|------|--------|----------------|
| **Decomposer** | Paper → Structured Claims | `list[Claim]` | Classifies into hypothesis/finding/method/limitation/future_work |
| **CrossReferencer** | Claim → Knowledge Graph | `list[Relation]` | 7 relation types across 7 paper boundaries |
| **GapAnalyzer** | Graph → Research Gaps | `list[ResearchGap]` | Structural density analysis + LLM semantic gap detection |
| **Synthesizer** | Claims+Graph → Narrative | `dict` | Consensus, tensions, idea evolution, meta-insights |
| **Critic** | Review → Quality Gate | `dict` | Methodology audit, statistical rigor, overstated claims |

### Long-Chain Reasoning in Action

Every agent produces an explicit `ReasoningChain` — a sequence of numbered steps showing *how* it reached each conclusion:

```
Question: "What are the main alignment approaches?"

Step 1 [Orchestrator/retrieve]
  Thought: Searching for claims relevant to "alignment approaches"
  Observation: Found 12 relevant claims across 5 papers

Step 2 [Orchestrator/graph_traverse]
  Thought: Tracing relations from relevant claims
  Observation: Expanded to 34 connected nodes via supports/contradicts edges

Step 3 [Orchestrator/synthesize]
  Thought: Synthesizing answer from retrieved claims and graph structure
  Conclusion: Three main approaches identified — RLHF (human feedback),
  DPO (direct optimization), and Constitutional AI (AI feedback) ...
```

All traces are stored in `AnalysisReport.reasoning_traces` for full auditability.

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/deepsurvey.git
cd deepsurvey

# Core dependencies (required)
pip install pydantic networkx

# Optional: real LLM backends
pip install openai        # OpenAI / compatible APIs
pip install anthropic     # Anthropic Claude
```

**Requirements:** Python 3.10+

---

## Quick Start

### Zero-Config Demo (no API keys needed)

```bash
python -m deepsurvey.main
```

Runs a complete analysis of 5 real AI Alignment papers (2022–2024) using the built-in simulation backend. You'll see:

- Structured claims extracted from each paper
- Cross-paper relations (supports, contradicts, extends)
- Research gaps with significance scores
- Synthesis report with consensus narrative and tensions
- Critical methodology review
- Knowledge graph statistics

### With a Real LLM

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
python -m deepsurvey.main --provider openai

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."
python -m deepsurvey.main --provider anthropic

# Any OpenAI-compatible endpoint (vLLM, Ollama, DeepSeek, etc.)
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="http://localhost:8000/v1"
python -m deepsurvey.main --provider openai --model "deepseek-v3"
```

### Ask a Question

```bash
python -m deepsurvey.main --query "What are the tradeoffs between RLHF and DPO?"
```

### Export the Report

```bash
python -m deepsurvey.main -o synthesis_report.json
```

---

## Programmatic Usage

```python
from deepsurvey.llm_client import LLMClient
from deepsurvey.models import Paper, Author
from deepsurvey.orchestrator import ResearchOrchestrator

# 1. Prepare your papers
papers = [
    Paper(
        id="paper_1",
        title="Your Paper Title",
        authors=[Author(name="First Author", affiliation="University")],
        year=2024,
        abstract="The paper investigates...",
        citations=["other_paper_id"],
        sections={
            "method": "We used...",
            "results": "We found...",
            "limitations": "Our approach assumes...",
        },
    ),
    # ... more papers
]

# 2. Choose a backend and run
llm = LLMClient(provider="simulation")  # or "openai" / "anthropic"
orchestrator = ResearchOrchestrator(llm=llm, max_iterations=3)

report = orchestrator.analyze(papers)

# 3. Explore the results
print(report.executive_summary)

for gap in report.research_gaps:
    print(f"[{gap.significance}] {gap.description}")
    print(f"  → Suggested: {gap.suggested_approach}")

# 4. Ask follow-up questions
answer = orchestrator.query("What evaluation benchmarks are missing?")
```

---

## Project Structure

```
deepsurvey/
├── __init__.py           # Package metadata
├── models.py             # 10 Pydantic data models (Paper, Claim, Relation, ...)
├── llm_client.py         # Multi-backend LLM with chain-of-thought support
├── knowledge_graph.py    # networkx-based academic knowledge graph
├── agents.py             # 5 specialized agents with CoT reasoning
├── orchestrator.py       # Multi-agent coordination + iterative refinement
└── main.py               # CLI entry point + built-in demo papers
```

### Data Models

| Model | Purpose |
|-------|---------|
| `Paper` | Research paper with metadata, abstract, sections, citations |
| `Claim` | Structured claim typed as hypothesis/finding/method/limitation/future_work |
| `Relation` | Typed directed edge: supports, contradicts, extends, cites, uses_method, addresses_limitation, motivates |
| `ResearchGap` | Identified gap with significance level and suggested approach |
| `Contradiction` | Conflict between two claims with severity and resolution notes |
| `ReasoningChain` | Full CoT trace as a sequence of `ReasoningStep` entries |
| `AnalysisReport` | Final synthesis aggregating all findings and traces |

### Knowledge Graph Capabilities

| Method | What it does |
|--------|-------------|
| `find_citation_chains()` | Multi-hop path tracing between any two nodes |
| `get_supporting_claims()` | Find all claims that support a given claim |
| `get_contradicting_claims()` | Find all claims that contradict a given claim |
| `get_transitive_relations()` | Compute transitive closure of any relation type |
| `detect_contradictions()` | Auto-detect all contradict edges in the graph |
| `identify_structural_gaps()` | Find isolated claims and weakly connected paper clusters |
| `neighborhood_summary()` | Human-readable summary of any node and its connections |

---

## Customization

### Custom Agent Prompt

```python
from deepsurvey.agents import DecomposerAgent

class DomainDecomposer(DecomposerAgent):
    def system_prompt(self) -> str:
        return (
            "You are an expert in biomedical NLP. Extract claims about: "
            "1) Drug interactions, 2) Clinical outcomes, 3) Side effects, "
            "4) Dosage effects. For each claim, cite the exact text span."
        )
```

### Custom LLM Backend

```python
llm = LLMClient(
    provider="openai",
    model="your-fine-tuned-model",
    api_key="sk-...",
    base_url="http://your-endpoint:8000/v1",
)
```

---

## Roadmap

- [ ] **PDF ingestion** — PyMuPDF-based parser with section structure detection
- [ ] **Embedding-based similarity** — semantic claim matching for more accurate cross-referencing
- [ ] **Web UI** — interactive knowledge graph exploration and report navigation
- [ ] **API integration** — auto-fetch papers from Semantic Scholar, arXiv, PubMed
- [ ] **Neo4j backend** — persistent graph storage for large-scale corpora (1000+ papers)
- [ ] **Multi-lingual** — support for Chinese, Japanese, Korean academic papers
- [ ] **Benchmark** — evaluation against human-written systematic reviews

---

## Citation

```bibtex
@software{deepsurvey2025,
  author = {DeepSurvey Contributors},
  title = {DeepSurvey: Deep Multi-Agent Literature Synthesis},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/deepsurvey},
  note = {An academic-oriented multi-agent system for automated literature review},
}
```

GitHub automatically recognizes `CITATION.cff` and displays a "Cite this repository" widget.

---

## License

MIT — See [LICENSE](LICENSE).
