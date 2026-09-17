"""Event-driven scheduling environment for training the RL policy.

One episode replays a real openb_* pod trace (data/csv/) against a real node
population in creation-time order. Each decision point is one pod that has
at least one node able to host it; the agent picks which of those candidate
nodes to place it on. Pods that later depart (deletion_time) free their
resources back onto their node, so the cluster genuinely fills, fragments,
and drains over an episode rather than monotonically filling up.

This is a Python-native re-implementation of the resource/fragmentation
bookkeeping in pkg/type/resource.go and pkg/utils, not a wrapper around the
Go binary: the Go simulator drives its scheduling loop autonomously through
the Kubernetes scheduler framework's Filter/Score/Bind pipeline and has no
"give me candidates, I'll pick one" API to call into step by step, and
shelling out to it per decision would be far too slow for the ~1e5-1e6 env
steps a PPO run needs anyway. Reward uses the exact fragmentation formula
ported in rlscore.frag, so it is measuring the same thing FGDScorePlugin
does in production, on the same trace data -- only the actor doing the
picking differs.
"""

import heapq
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from rlscore import trace
from rlscore.features import encode
from rlscore.frag import (
    MILLI,
    NodeState,
    PodShape,
    can_node_host_pod_on_gpu_memory,
    is_node_accessible_to_pod,
    node_gpu_share_frag_score,
)

FAILURE_REWARD = -5.0  # a pod that no node could fit is dropped
FRAG_REWARD_SCALE = 1.0 / MILLI  # keeps rewards in a friendly O(1) range


@dataclass
class Placement:
    node_index: int
    milli_cpu: float
    gpu_allocations: List[Tuple[int, float]]  # (gpu_index, milli_gpu_reserved)


@dataclass
class Observation:
    pod: PodShape
    candidate_node_indices: List[int]  # indices into env.nodes
    features: torch.Tensor  # [len(candidate_node_indices), FEATURE_DIM]


def _candidate_indices(nodes: List[NodeState], pod: PodShape) -> List[int]:
    out = []
    for i, node in enumerate(nodes):
        if node.milli_cpu_left < pod.milli_cpu:
            continue
        if pod.milli_gpu > 0:
            if not is_node_accessible_to_pod(node.gpu_type, pod.gpu_type):
                continue
            if not can_node_host_pod_on_gpu_memory(node, pod):
                continue
        out.append(i)
    return out


class SchedulingEnv:
    def __init__(self, node_csv_path: str, pod_csv_path: str, max_pods: Optional[int] = None):
        self._node_template = trace.load_nodes(node_csv_path)
        self._pods = trace.load_pods(pod_csv_path, max_pods=max_pods)
        self.typical_pods = trace.typical_pod_distribution(self._pods)
        self.total_pods = len(self._pods)

        self.nodes: List[NodeState] = []
        self._departures: List[Tuple[float, int]] = []  # heap of (deletion_time, pod_idx)
        self._placements = {}
        self._pod_idx = 0
        self._current: Optional[Observation] = None
        self.placed = 0
        self.dropped = 0

    def reset(self) -> Observation:
        self.nodes = [
            NodeState(n.name, n.milli_cpu_capacity, n.milli_cpu_capacity, list(n.milli_gpu_left_list), n.gpu_type)
            for n in self._node_template
        ]
        self._departures = []
        self._placements = {}
        self._pod_idx = 0
        self.placed = 0
        self.dropped = 0
        obs, _, _ = self._advance()
        self._current = obs
        return obs

    def _apply_departures_up_to(self, t: float) -> None:
        while self._departures and self._departures[0][0] <= t:
            _, pod_idx = heapq.heappop(self._departures)
            placement = self._placements.pop(pod_idx, None)
            if placement is None:
                continue
            node = self.nodes[placement.node_index]
            node.milli_cpu_left += placement.milli_cpu
            for gpu_idx, amount in placement.gpu_allocations:
                node.milli_gpu_left_list[gpu_idx] += amount

    def _advance(self) -> Tuple[Optional[Observation], float, bool]:
        """Skip forward, auto-dropping pods with no feasible node, until the
        next pod with >=1 candidate (or the trace is exhausted)."""
        accumulated_reward = 0.0
        while self._pod_idx < self.total_pods:
            pod = self._pods[self._pod_idx]
            self._apply_departures_up_to(pod.creation_time)
            candidates = _candidate_indices(self.nodes, pod.shape)
            if candidates:
                features = torch.tensor(
                    [encode(self.nodes[i], pod.shape) for i in candidates], dtype=torch.float32
                )
                return Observation(pod.shape, candidates, features), accumulated_reward, False
            self.dropped += 1
            accumulated_reward += FAILURE_REWARD
            self._pod_idx += 1
        return None, accumulated_reward, True

    def _apply_placement(self, node_index: int, pod: PodShape) -> Placement:
        node = self.nodes[node_index]
        node.milli_cpu_left -= pod.milli_cpu

        gpu_allocations: List[Tuple[int, float]] = []
        if pod.milli_gpu > 0:
            if pod.gpu_number > 1 or pod.milli_gpu >= MILLI:
                # exclusive: consume `gpu_number` fully-free GPUs
                chosen = [i for i, g in enumerate(node.milli_gpu_left_list) if g >= MILLI][: pod.gpu_number]
                for gpu_idx in chosen:
                    node.milli_gpu_left_list[gpu_idx] -= MILLI
                    gpu_allocations.append((gpu_idx, MILLI))
            else:
                # fractional share on a single GPU: best-fit (smallest sufficient slot)
                sufficient = [(g, i) for i, g in enumerate(node.milli_gpu_left_list) if g >= pod.milli_gpu]
                _, gpu_idx = min(sufficient)
                node.milli_gpu_left_list[gpu_idx] -= pod.milli_gpu
                gpu_allocations.append((gpu_idx, pod.milli_gpu))

        return Placement(node_index, pod.milli_cpu, gpu_allocations)

    def step(self, action: int) -> Tuple[Optional[Observation], float, bool, dict]:
        """action indexes into self._current.candidate_node_indices (NOT a raw node index)."""
        if self._current is None:
            raise RuntimeError("step() called with no pending decision -- call reset() or check `done`")

        node_index = self._current.candidate_node_indices[action]
        pod = self._current.pod
        node = self.nodes[node_index]

        frag_before = node_gpu_share_frag_score(node, self.typical_pods)
        placement = self._apply_placement(node_index, pod)
        frag_after = node_gpu_share_frag_score(node, self.typical_pods)
        reward = (frag_before - frag_after) * FRAG_REWARD_SCALE

        self._placements[self._pod_idx] = placement
        if self._current.pod is not None:
            current_pod_record = self._pods[self._pod_idx]
            if current_pod_record.deletion_time < float("inf"):
                heapq.heappush(self._departures, (current_pod_record.deletion_time, self._pod_idx))
        self.placed += 1
        self._pod_idx += 1

        next_obs, extra_reward, done = self._advance()
        self._current = next_obs
        return next_obs, reward + extra_reward, done, {"placed": self.placed, "dropped": self.dropped}
