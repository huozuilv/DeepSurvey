"""
Academic knowledge graph with multi-hop reasoning capabilities.
Stores papers, claims, and relations; supports graph traversal,
citation chain tracing, contradiction detection, and gap analysis.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

import networkx as nx

from .models import (
    Claim,
    ClaimType,
    Contradiction,
    Paper,
    Relation,
    RelationType,
    ResearchGap,
)


class KnowledgeGraph:
    """
    A directed knowledge graph for academic research artifacts.
    Nodes = Papers + Claims. Edges = Relations.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._papers: dict[str, Paper] = {}
        self._claims: dict[str, Claim] = {}
        self._relations: dict[str, Relation] = {}

    # ── Node Management ──────────────────────────────────────────────────────

    def add_paper(self, paper: Paper) -> str:
        self._papers[paper.id] = paper
        self.graph.add_node(paper.id, kind="paper", label=paper.title[:80])
        return paper.id

    def add_claim(self, claim: Claim) -> str:
        self._claims[claim.id] = claim
        self.graph.add_node(
            claim.id,
            kind="claim",
            type=claim.type.value,
            label=claim.content[:80],
        )
        # Auto-link claim to its paper
        if claim.paper_id in self._papers:
            self.graph.add_edge(
                claim.paper_id,
                claim.id,
                kind="contains",
            )
        return claim.id

    def add_relation(self, relation: Relation) -> str:
        self._relations[relation.id] = relation
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            kind=relation.type.value,
            explanation=relation.explanation,
            confidence=relation.confidence.value,
        )
        return relation.id

    # ── Queries ──────────────────────────────────────────────────────────────

    @property
    def paper_count(self) -> int:
        return len(self._papers)

    @property
    def claim_count(self) -> int:
        return len(self._claims)

    @property
    def relation_count(self) -> int:
        return len(self._relations)

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        return self._papers.get(paper_id)

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self._claims.get(claim_id)

    def get_claims_by_type(self, claim_type: ClaimType) -> list[Claim]:
        return [c for c in self._claims.values() if c.type == claim_type]

    def get_claims_by_paper(self, paper_id: str) -> list[Claim]:
        return [c for c in self._claims.values() if c.paper_id == paper_id]

    def get_relations_by_type(self, rel_type: RelationType) -> list[Relation]:
        return [r for r in self._relations.values() if r.type == rel_type]

    # ── Multi-hop Reasoning ──────────────────────────────────────────────────

    def find_citation_chains(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5,
    ) -> list[list[str]]:
        """Find all paths between two nodes (citation chain tracing)."""
        try:
            paths = list(
                nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_length)
            )
            return paths
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_supporting_claims(self, claim_id: str) -> list[Claim]:
        """Find claims that support the given claim."""
        supporting = []
        for pred in self.graph.predecessors(claim_id):
            edge = self.graph.edges.get((pred, claim_id), {})
            if edge.get("kind") == RelationType.SUPPORTS.value:
                if pred in self._claims:
                    supporting.append(self._claims[pred])
        return supporting

    def get_contradicting_claims(self, claim_id: str) -> list[Claim]:
        """Find claims that contradict the given claim."""
        contradicting = []
        for pred in self.graph.predecessors(claim_id):
            edge = self.graph.edges.get((pred, claim_id), {})
            if edge.get("kind") == RelationType.CONTRADICTS.value:
                if pred in self._claims:
                    contradicting.append(self._claims[pred])
        return contradicting

    def get_transitive_relations(
        self, source_id: str, relation_type: RelationType
    ) -> list[list[str]]:
        """Find transitive closure of a relation type starting from a node."""
        paths = []
        visited = set()

        def dfs(node: str, current_path: list[str]):
            visited.add(node)
            for succ in self.graph.successors(node):
                edge = self.graph.edges.get((node, succ), {})
                if edge.get("kind") == relation_type.value:
                    new_path = current_path + [succ]
                    paths.append(new_path)
                    if succ not in visited:
                        dfs(succ, new_path)

        dfs(source_id, [source_id])
        return paths

    # ── Contradiction Detection ──────────────────────────────────────────────

    def detect_contradictions(self) -> list[Contradiction]:
        """Detect contradictions by traversing the graph for contradict edges."""
        contradictions = []
        for source, target, data in self.graph.edges(data=True):
            if data.get("kind") == RelationType.CONTRADICTS.value:
                claim_a = self._claims.get(source)
                claim_b = self._claims.get(target)
                if claim_a and claim_b:
                    contradictions.append(
                        Contradiction(
                            claim_a_id=source,
                            claim_b_id=target,
                            description=f"'{claim_a.content[:60]}...' vs '{claim_b.content[:60]}...'",
                            resolution=data.get("explanation", ""),
                        )
                    )
        return contradictions

    # ── Gap Analysis ─────────────────────────────────────────────────────────

    def identify_structural_gaps(self) -> list[ResearchGap]:
        """Identify structural gaps based on graph density analysis."""
        gaps = []

        # 1. Identify isolated claims (no relations)
        for claim_id, claim in self._claims.items():
            in_degree = self.graph.in_degree(claim_id)
            out_degree = self.graph.out_degree(claim_id)
            if in_degree == 0 and out_degree == 0:
                gaps.append(
                    ResearchGap(
                        description=f"Claim '{claim.content[:60]}...' has no connections — potential unexplored area",
                        related_claims=[claim_id],
                        related_papers=[claim.paper_id],
                        significance="medium",
                    )
                )

        # 2. Identify under-connected paper clusters
        paper_nodes = [pid for pid in self._papers]
        if len(paper_nodes) >= 2:
            for i in range(len(paper_nodes)):
                for j in range(i + 1, len(paper_nodes)):
                    try:
                        path = nx.shortest_path(self.graph, paper_nodes[i], paper_nodes[j])
                        if len(path) > 4:  # Long path = weak connection
                            gaps.append(
                                ResearchGap(
                                    description=f"Weak connection between papers '{self._papers[paper_nodes[i]].title[:40]}' and '{self._papers[paper_nodes[j]].title[:40]}'",
                                    related_papers=[paper_nodes[i], paper_nodes[j]],
                                    significance="low",
                                    suggested_approach="Investigate potential cross-fertilization between these areas",
                                )
                            )
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        # No path exists at all — stronger gap signal
                        gaps.append(
                            ResearchGap(
                                description=f"No connection between papers '{self._papers[paper_nodes[i]].title[:40]}' and '{self._papers[paper_nodes[j]].title[:40]}' — potential bridging research needed",
                                related_papers=[paper_nodes[i], paper_nodes[j]],
                                significance="high",
                                suggested_approach="Explore whether findings from one area apply to the other",
                            )
                        )

        return gaps

    # ── Graph Statistics ─────────────────────────────────────────────────────

    def get_statistics(self) -> dict:
        """Compute graph statistics for reporting."""
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "papers": self.paper_count,
            "claims": self.claim_count,
            "relations": self.relation_count,
        }

        # Claim type distribution
        claim_types = defaultdict(int)
        for c in self._claims.values():
            claim_types[c.type.value] += 1
        stats["claim_types"] = dict(claim_types)

        # Relation type distribution
        rel_types = defaultdict(int)
        for r in self._relations.values():
            rel_types[r.type.value] += 1
        stats["relation_types"] = dict(rel_types)

        # Density
        n = stats["total_nodes"]
        if n > 1:
            stats["density"] = self.graph.number_of_edges() / (n * (n - 1))
        else:
            stats["density"] = 0.0

        # Connected components
        undirected = self.graph.to_undirected()
        stats["connected_components"] = nx.number_connected_components(undirected)

        return stats

    # ── Export ───────────────────────────────────────────────────────────────

    def summarize_node(self, node_id: str) -> str:
        """Human-readable summary of a node."""
        if node_id in self._papers:
            p = self._papers[node_id]
            return f"[Paper] {p.title} ({p.year}) — {p.authors[0].name if p.authors else 'Unknown'}"
        if node_id in self._claims:
            c = self._claims[node_id]
            return f"[Claim:{c.type.value}] {c.content[:100]}"
        return f"[Unknown] {node_id}"

    def neighborhood_summary(self, node_id: str, radius: int = 1) -> str:
        """Summarize the neighborhood of a node."""
        if node_id not in self.graph:
            return f"Node {node_id} not found."

        lines = [f"Node: {self.summarize_node(node_id)}", ""]

        # Incoming
        incoming = list(self.graph.in_edges(node_id, data=True))
        if incoming:
            lines.append("Incoming relations:")
            for src, _, data in incoming:
                lines.append(f"  ← {data.get('kind', '?')} — {self.summarize_node(src)}")

        # Outgoing
        outgoing = list(self.graph.out_edges(node_id, data=True))
        if outgoing:
            lines.append("Outgoing relations:")
            for _, tgt, data in outgoing:
                lines.append(f"  → {data.get('kind', '?')} — {self.summarize_node(tgt)}")

        return "\n".join(lines)
