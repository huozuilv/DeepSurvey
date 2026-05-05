"""
DeepSurvey — Deep Multi-Agent Literature Synthesis
====================================================
Academic-oriented multi-agent system for automated literature review,
knowledge graph construction, and research gap analysis.

Features:
  - Long-chain reasoning: Each agent produces explicit reasoning traces
  - Multi-agent collaboration: 5 specialized agents with message passing
  - Knowledge graph: Structured representation of claims and relations
  - Gap analysis: Automated identification of under-explored research areas
  - Interactive query: Ask questions about the analyzed literature

Usage:
  python -m deepsurvey.main          # Run with demo papers
  python -m deepsurvey.main --help   # Show options
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from typing import List

from .llm_client import LLMClient
from .models import Author, Paper
from .orchestrator import ResearchOrchestrator


# ── Demo Papers ──────────────────────────────────────────────────────────────

def create_demo_papers() -> List[Paper]:
    """
    Create a set of realistic demo papers about Large Language Model
    alignment and safety — a topic with genuine academic debates,
    allowing the system to demonstrate contradiction detection and
    gap analysis.
    """
    return [
        Paper(
            id="paper_rlhf_2022",
            title="Training Language Models to Follow Instructions with Human Feedback",
            authors=[
                Author(name="Long Ouyang", affiliation="OpenAI"),
                Author(name="Jeff Wu", affiliation="OpenAI"),
                Author(name="Ryan Lowe", affiliation="OpenAI"),
            ],
            year=2022,
            venue="NeurIPS 2022",
            abstract=(
                "We present InstructGPT, a method for aligning language models with user intent "
                "through reinforcement learning from human feedback (RLHF). We collect human "
                "preference data comparing model outputs and use it to train a reward model, "
                "then fine-tune GPT-3 via PPO. Results show InstructGPT produces outputs preferred "
                "over GPT-3 85% of the time while improving truthfulness and reducing toxicity. "
                "However, the approach requires significant human annotation effort and may not "
                "scale to all cultural contexts."
            ),
            citations=["paper_dpo_2023", "paper_rlaif_2023"],
            sections={
                "introduction": "Aligning language models with human values is critical for safe deployment.",
                "method": "RLHF with PPO optimization and human preference-based reward model.",
                "results": "85% win rate against base GPT-3; improved truthfulness by 21%.",
                "limitations": "Human annotation cost; cultural bias in English-only preference data.",
            },
        ),
        Paper(
            id="paper_dpo_2023",
            title="Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
            authors=[
                Author(name="Rafael Rafailov", affiliation="Stanford University"),
                Author(name="Archit Sharma", affiliation="Stanford University"),
            ],
            year=2023,
            venue="NeurIPS 2023",
            abstract=(
                "We propose Direct Preference Optimization (DPO), which eliminates the need for "
                "a separate reward model in RLHF. DPO reparameterizes the reward function in "
                "terms of the policy, enabling direct optimization from preference data. "
                "On controlled sentiment generation and summarization tasks, DPO matches or "
                "exceeds RLHF performance while being simpler and more stable. However, DPO "
                "still requires pairwise preference data, which may not capture complex value "
                "tradeoffs, and the theoretical guarantees assume the Bradley-Terry preference model."
            ),
            citations=["paper_rlhf_2022"],
            sections={
                "introduction": "RLHF requires training a reward model separately, adding complexity.",
                "method": "Direct optimization from preferences without explicit reward modeling.",
                "results": "Matches RLHF on summarization; exceeds on sentiment control.",
                "limitations": "Bradley-Terry model assumption; binary preference limitation.",
            },
        ),
        Paper(
            id="paper_rlaif_2023",
            title="Constitutional AI: Harmlessness from AI Feedback",
            authors=[
                Author(name="Yuntao Bai", affiliation="Anthropic"),
                Author(name="Saurav Kadavath", affiliation="Anthropic"),
            ],
            year=2023,
            venue="arXiv preprint",
            abstract=(
                "We introduce Constitutional AI (CAI), which replaces human feedback with AI-generated "
                "feedback guided by a constitution of principles. CAI uses chain-of-thought reasoning "
                "to evaluate model outputs against constitutional rules, then trains via both "
                "supervised fine-tuning and RL. Results show CAI-trained models are both helpful "
                "and harmless, with harmlessness scores comparable to human-feedback-trained models. "
                "A key advantage is scalability beyond human annotation bandwidth, though the "
                "constitution itself may encode designer biases."
            ),
            citations=["paper_rlhf_2022", "paper_dpo_2023"],
            sections={
                "introduction": "Human feedback is expensive and hard to scale.",
                "method": "AI feedback guided by constitutional principles with CoT evaluation.",
                "results": "Comparable harmlessness to RLHF; better scalability.",
                "limitations": "Constitution design encodes biases; AI feedback may miss nuanced harms.",
            },
        ),
        Paper(
            id="paper_mech_interp_2024",
            title="Towards Monosemanticity: Decomposing Language Models With Dictionary Learning",
            authors=[
                Author(name="Trenton Bricken", affiliation="Anthropic"),
                Author(name="Adly Templeton", affiliation="Anthropic"),
            ],
            year=2024,
            venue="ICML 2024",
            abstract=(
                "We apply sparse dictionary learning to decompose transformer MLP activations "
                "into interpretable monosemantic features. Using a sparse autoencoder, we extract "
                "features that correspond to human-understandable concepts (DNA sequences, legal "
                "language, mathematical operations). We demonstrate that many features previously "
                "thought polysemantic can be decomposed. Limitations include the computational "
                "cost of training sparse autoencoders at scale and the fact that only a subset "
                "of features are interpretable — many remain opaque. This work opens the door "
                "to mechanistic interpretability at scale."
            ),
            citations=["paper_rlhf_2022"],
            sections={
                "introduction": "Understanding internal model representations is key to alignment.",
                "method": "Sparse dictionary learning on MLP activations using autoencoders.",
                "results": "Decomposed ~50K features; many correspond to interpretable concepts.",
                "limitations": "Computationally expensive; many features remain uninterpretable.",
            },
        ),
        Paper(
            id="paper_scalable_oversight_2024",
            title="Scalable Oversight of AI Systems via Debate",
            authors=[
                Author(name="Geoffrey Irving", affiliation="DeepMind"),
                Author(name="Paul Christiano", affiliation="Alignment Research Center"),
            ],
            year=2024,
            venue="Science",
            abstract=(
                "We formalize AI debate as a method for scalable oversight where two AI systems "
                "argue opposing sides of a question while a human judge determines the truth. "
                "We prove that under certain assumptions, debate can elicit truthful answers even "
                "when individual AIs are more capable than the human judge. Empirical results on "
                "reading comprehension and image classification tasks show debate improves human "
                "judgment accuracy by 15-25%. However, the approach assumes the question has a "
                "definable truth, which may not hold for value-laden questions, and the human "
                "judge must be sufficiently attentive to follow multi-turn arguments."
            ),
            citations=["paper_rlhf_2022", "paper_rlaif_2023"],
            sections={
                "introduction": "As AI surpasses human capabilities, direct oversight becomes impossible.",
                "method": "Adversarial debate between AIs with human as judge.",
                "results": "15-25% improvement in human judgment accuracy.",
                "limitations": "Assumes definable truth; requires attentive human judges.",
            },
        ),
    ]


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DeepSurvey — Deep Multi-Agent Literature Synthesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m deepsurvey.main                    Run demo with built-in papers
  python -m deepsurvey.main --demo             Same as above
  python -m deepsurvey.main --papers papers.json  Analyze papers from JSON file
  python -m deepsurvey.main --query "What are the main alignment approaches?"
        """,
    )
    parser.add_argument(
        "--demo", action="store_true", default=True,
        help="Run with built-in demo papers (default)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress detailed progress output",
    )
    parser.add_argument(
        "--query", type=str, default="",
        help="Interactive question to ask after analysis",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="",
        help="Save report to JSON file",
    )
    parser.add_argument(
        "--provider", type=str, default="simulation",
        help="LLM provider: openai, anthropic, or simulation (default)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3,
        help="Maximum refinement iterations (default: 3)",
    )

    args = parser.parse_args()

    # Create LLM client
    llm = LLMClient(provider=args.provider)

    # Load papers
    if args.demo:
        papers = create_demo_papers()
        print(f"\n{'#'*70}")
        print(f"  DeepSurvey — Deep Multi-Agent Literature Synthesis")
        print(f"  Academic Literature Review & Analysis System")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  LLM Backend: {llm.provider} ({llm.model})")
        print(f"{'#'*70}")
        print(f"\n  Demo Topic: AI Alignment & Safety (2022-2024)")
        print(f"  Papers loaded: {len(papers)}")
        for p in papers:
            print(f"    • {p.title[:70]}... ({p.year}, {p.venue})")
    else:
        print("Custom paper loading not yet implemented. Use --demo.")
        sys.exit(1)

    # Run the multi-agent pipeline
    orchestrator = ResearchOrchestrator(
        llm=llm,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )

    report = orchestrator.analyze(papers)

    # Display report
    print(f"\n{'='*70}")
    print(f"  SYNTHESIS REPORT")
    print(f"{'='*70}")

    print(f"\n📋 Executive Summary:")
    print(f"   {report.executive_summary[:500] or '(See detailed synthesis below)'}")

    print(f"\n🔑 Key Findings ({len(report.key_findings)}):")
    for i, f in enumerate(report.key_findings[:8], 1):
        print(f"   {i}. {f[:120]}")

    print(f"\n⚠️  Contradictions Detected ({len(report.contradictions)}):")
    for c in report.contradictions:
        print(f"   • {c.description[:120]}")

    print(f"\n🕳️  Research Gaps ({len(report.research_gaps)}):")
    for g in report.research_gaps[:5]:
        sig = f"[{g.significance.upper()}]" if hasattr(g, 'significance') else ""
        print(f"   • {sig} {g.description[:120]}")
        if hasattr(g, 'suggested_approach') and g.suggested_approach:
            print(f"     → {g.suggested_approach[:120]}")

    print(f"\n🔮 Future Directions:")
    for d in report.future_directions[:6]:
        print(f"   • {d[:120]}")

    print(f"\n📊 Knowledge Graph Statistics:")
    for k, v in report.knowledge_graph_stats.items():
        print(f"   {k}: {v}")

    print(f"\n🧠 Reasoning Chains: {len(report.reasoning_traces)} chains, "
          f"{sum(len(c.steps) for c in report.reasoning_traces)} total steps")

    # Interactive query
    if args.query:
        print(f"\n{'='*70}")
        print(f"  QUERY: {args.query}")
        print(f"{'='*70}")
        answer = orchestrator.query(args.query)
        print(f"\n{answer}")

    # Save report
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(), f, indent=2, default=str, ensure_ascii=False)
        print(f"\n📁 Report saved to: {args.output}")

    print(f"\n{'='*70}")
    print(f"  Analysis complete. {len(papers)} papers, "
          f"{report.knowledge_graph_stats.get('claims', 0)} claims, "
          f"{report.knowledge_graph_stats.get('relations', 0)} relations.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
