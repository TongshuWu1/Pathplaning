"""Minimal CAGE route graph and route certificates."""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .geometry import Point, distance, segment_length


@dataclass
class GraphNode:
    id: int
    xy: Point
    kind: str
    confidence: float = 1.0


@dataclass
class EdgeCertificate:
    confidence: float
    length: float
    min_clearance: float
    mean_consistency: float
    pose_quality: float
    traversal_success: int = 0
    failed_traversal: int = 0
    source_robots: set[int] = field(default_factory=set)
    last_updated: float = 0.0
    reported_home: bool = False

    def update(self, clearance: float, consistency: float, pose_quality: float, robot_id: int, time_s: float, success: bool = True) -> None:
        self.min_clearance = min(self.min_clearance, clearance) if self.min_clearance > 0 else clearance
        self.mean_consistency = 0.7 * self.mean_consistency + 0.3 * consistency
        self.pose_quality = 0.7 * self.pose_quality + 0.3 * pose_quality
        if success:
            self.traversal_success += 1
        else:
            self.failed_traversal += 1
        self.source_robots.add(robot_id)
        self.last_updated = time_s
        self.confidence = compute_edge_confidence(
            min_clearance=self.min_clearance,
            consistency=self.mean_consistency,
            pose_quality=self.pose_quality,
            traversal_success=self.traversal_success,
            failed_traversal=self.failed_traversal,
        )


@dataclass
class GraphEdge:
    id: int
    a: int
    b: int
    cert: EdgeCertificate


@dataclass
class RouteCandidate:
    node_ids: list[int]
    edge_ids: list[int]
    length: float
    min_clearance: float
    certificate: float
    reported_home: bool
    status: str


def compute_edge_confidence(
    min_clearance: float,
    consistency: float,
    pose_quality: float,
    traversal_success: int,
    failed_traversal: int,
) -> float:
    clear_score = max(0.05, min(1.0, (min_clearance - 0.15) / 0.85))
    trav_score = min(1.0, 0.35 + 0.18 * max(0, traversal_success))
    fail_penalty = 0.22 * failed_traversal
    raw = 0.34 * clear_score + 0.25 * consistency + 0.18 * pose_quality + 0.23 * trav_score - fail_penalty
    return float(max(0.0, min(1.0, raw)))


class RouteGraph:
    def __init__(self, merge_distance: float = 0.55):
        self.merge_distance = merge_distance
        self.nodes: dict[int, GraphNode] = {}
        self.edges: dict[int, GraphEdge] = {}
        self._adj: dict[int, dict[int, int]] = {}
        self._next_node_id = 0
        self._next_edge_id = 0
        self.home_id: int | None = None
        self.target_id: int | None = None
        self._version = 0
        self._route_cache_key: tuple[int, int, bool, int | None, int | None] | None = None
        self._route_cache: list[RouteCandidate] = []

    def _touch(self) -> None:
        self._version += 1
        self._route_cache_key = None
        self._route_cache = []

    def copy(self) -> "RouteGraph":
        other = RouteGraph(self.merge_distance)
        other.nodes = {i: GraphNode(n.id, n.xy, n.kind, n.confidence) for i, n in self.nodes.items()}
        other.edges = {
            i: GraphEdge(e.id, e.a, e.b, EdgeCertificate(
                confidence=e.cert.confidence,
                length=e.cert.length,
                min_clearance=e.cert.min_clearance,
                mean_consistency=e.cert.mean_consistency,
                pose_quality=e.cert.pose_quality,
                traversal_success=e.cert.traversal_success,
                failed_traversal=e.cert.failed_traversal,
                source_robots=set(e.cert.source_robots),
                last_updated=e.cert.last_updated,
                reported_home=e.cert.reported_home,
            )) for i, e in self.edges.items()
        }
        other._adj = {a: dict(bs) for a, bs in self._adj.items()}
        other._next_node_id = self._next_node_id
        other._next_edge_id = self._next_edge_id
        other.home_id = self.home_id
        other.target_id = self.target_id
        other._version = self._version
        return other

    def add_node(self, xy: Point, kind: str = "keypoint", confidence: float = 1.0, allow_merge: bool = True) -> int:
        if allow_merge:
            merge_distance = max(self.merge_distance, 1.6) if kind == "target" else self.merge_distance
            for nid, node in self.nodes.items():
                if node.kind == kind and distance(node.xy, xy) <= merge_distance:
                    if confidence > node.confidence:
                        node.confidence = confidence
                        self._touch()
                    return nid
            # Keypoints can merge into anchors/home/target if very close.
            for nid, node in self.nodes.items():
                if distance(node.xy, xy) <= self.merge_distance * 0.55:
                    if confidence > node.confidence:
                        node.confidence = confidence
                        self._touch()
                    return nid
        nid = self._next_node_id
        self._next_node_id += 1
        self.nodes[nid] = GraphNode(nid, xy, kind, confidence)
        self._adj.setdefault(nid, {})
        if kind == "home":
            self.home_id = nid
        elif kind == "target":
            self.target_id = nid
        self._touch()
        return nid

    def add_or_update_edge(
        self,
        a: int,
        b: int,
        clearance: float,
        consistency: float,
        pose_quality: float,
        robot_id: int,
        time_s: float,
        success: bool = True,
    ) -> int | None:
        if a == b or a not in self.nodes or b not in self.nodes:
            return None
        if b in self._adj.get(a, {}):
            eid = self._adj[a][b]
            self.edges[eid].cert.update(clearance, consistency, pose_quality, robot_id, time_s, success)
            self._touch()
            return eid
        length = distance(self.nodes[a].xy, self.nodes[b].xy)
        conf = compute_edge_confidence(clearance, consistency, pose_quality, int(success), int(not success))
        cert = EdgeCertificate(
            confidence=conf,
            length=length,
            min_clearance=clearance,
            mean_consistency=consistency,
            pose_quality=pose_quality,
            traversal_success=int(success),
            failed_traversal=int(not success),
            source_robots={robot_id},
            last_updated=time_s,
        )
        eid = self._next_edge_id
        self._next_edge_id += 1
        self.edges[eid] = GraphEdge(eid, a, b, cert)
        self._adj.setdefault(a, {})[b] = eid
        self._adj.setdefault(b, {})[a] = eid
        self._touch()
        return eid

    def merge_from_digest(self, digest: dict) -> None:
        id_map: dict[int, int] = {}
        for nd in digest.get("nodes", []):
            nid_old = int(nd["id"])
            nid_new = self.add_node(tuple(nd["xy"]), str(nd.get("kind", "keypoint")), float(nd.get("confidence", 1.0)))
            id_map[nid_old] = nid_new
        for ed in digest.get("edges", []):
            a = id_map.get(int(ed["a"]))
            b = id_map.get(int(ed["b"]))
            if a is None or b is None:
                continue
            cert_in = ed.get("cert", {})
            eid = self.add_or_update_edge(
                a, b,
                clearance=float(cert_in.get("min_clearance", 0.4)),
                consistency=float(cert_in.get("mean_consistency", 0.5)),
                pose_quality=float(cert_in.get("pose_quality", 0.5)),
                robot_id=int(digest.get("source_robot", -1)),
                time_s=float(digest.get("time_s", 0.0)),
                success=cert_in.get("failed_traversal", 0) == 0,
            )
            if eid is not None:
                edge = self.edges[eid]
                prev_conf = edge.cert.confidence
                prev_reported = edge.cert.reported_home
                edge.cert.confidence = max(edge.cert.confidence, float(cert_in.get("confidence", 0.0)))
                edge.cert.reported_home = edge.cert.reported_home or bool(cert_in.get("reported_home", False))
                if edge.cert.confidence != prev_conf or edge.cert.reported_home != prev_reported:
                    self._touch()

    def make_digest(self, robot_id: int, time_s: float, max_edges: int = 80) -> dict:
        # Send recent/high-confidence route evidence, always preserving route-critical edges.
        required: dict[int, GraphEdge] = {}
        for node_id in (self.home_id, self.target_id):
            if node_id is None:
                continue
            for eid in self._adj.get(node_id, {}).values():
                required[eid] = self.edges[eid]
        for route in self.top_routes(k=1, require_target=False):
            for eid in route.edge_ids:
                if eid in self.edges:
                    required[eid] = self.edges[eid]
        ranked = sorted(self.edges.values(), key=lambda e: (e.cert.confidence, e.cert.last_updated), reverse=True)
        edge_map = dict(required)
        for edge in ranked:
            if len(edge_map) >= max_edges:
                break
            edge_map.setdefault(edge.id, edge)
        edges = list(edge_map.values())
        node_ids = sorted({n for e in edges for n in (e.a, e.b)})
        return {
            "source_robot": int(robot_id),
            "time_s": float(time_s),
            "nodes": [
                {"id": int(nid), "xy": [float(self.nodes[nid].xy[0]), float(self.nodes[nid].xy[1])], "kind": self.nodes[nid].kind, "confidence": float(self.nodes[nid].confidence)}
                for nid in node_ids
            ],
            "edges": [
                {
                    "id": int(e.id),
                    "a": int(e.a),
                    "b": int(e.b),
                    "cert": {
                        "confidence": float(e.cert.confidence),
                        "length": float(e.cert.length),
                        "min_clearance": float(e.cert.min_clearance),
                        "mean_consistency": float(e.cert.mean_consistency),
                        "pose_quality": float(e.cert.pose_quality),
                        "traversal_success": int(e.cert.traversal_success),
                        "failed_traversal": int(e.cert.failed_traversal),
                        "reported_home": bool(e.cert.reported_home),
                    },
                }
                for e in edges
            ],
        }

    def mark_all_reported_home(self) -> None:
        changed = False
        for edge in self.edges.values():
            if not edge.cert.reported_home:
                edge.cert.reported_home = True
                changed = True
        if changed:
            self._touch()

    def top_routes(self, k: int = 3, require_target: bool = True) -> list[RouteCandidate]:
        cache_key = (self._version, k, require_target, self.home_id, self.target_id)
        if self._route_cache_key == cache_key:
            return list(self._route_cache)
        if self.home_id is None or self.target_id is None:
            return []
        if self.home_id not in self.nodes or self.target_id not in self.nodes:
            return []
        # Dijkstra-style best route search. The route graph can be a long chain,
        # so a small simple-path depth cap will miss valid H->target routes.
        pq: list[tuple[float, int, int, list[int], list[int], float, float, float, bool]] = []
        counter = 0
        heapq.heappush(pq, (0.0, counter, self.home_id, [self.home_id], [], 0.0, math.inf, 1.0, True))
        best_score: dict[int, float] = {self.home_id: 0.0}
        while pq and counter < max(2000, len(self.edges) * 40):
            score, _, cur, path_nodes, path_edges, length, min_clearance, cert, reported = heapq.heappop(pq)
            if score > best_score.get(cur, math.inf) + 1e-9:
                continue
            if cur == self.target_id:
                status = "certified" if cert >= 0.62 else "candidate"
                if not reported:
                    status += "/needs_report"
                routes = [RouteCandidate(path_nodes, path_edges, length, min_clearance if math.isfinite(min_clearance) else 0.0, cert, reported, status)]
                self._route_cache_key = cache_key
                self._route_cache = routes
                return list(routes)
            for nb, eid in self._adj.get(cur, {}).items():
                if nb in path_nodes:
                    continue
                edge = self.edges[eid]
                new_length = length + edge.cert.length
                new_min_clearance = min(min_clearance, edge.cert.min_clearance)
                new_cert = min(cert, edge.cert.confidence)
                new_reported = reported and edge.cert.reported_home
                # Prefer high confidence and short paths.
                new_score = new_length / max(new_cert, 0.05) + 3.0 * max(0.0, 0.65 - new_cert)
                if new_score >= best_score.get(nb, math.inf):
                    continue
                best_score[nb] = new_score
                counter += 1
                heapq.heappush(pq, (new_score, counter, nb, path_nodes + [nb], path_edges + [eid], new_length, new_min_clearance, new_cert, new_reported))
        self._route_cache_key = cache_key
        self._route_cache = []
        return []

    def route_points(self, route: RouteCandidate) -> list[Point]:
        return [self.nodes[n].xy for n in route.node_ids if n in self.nodes]
