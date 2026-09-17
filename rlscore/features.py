"""Encodes a (pod, candidate node) pair into a fixed-size feature vector.

Used identically by rlscore.env (training, built from CSV trace rows) and
rlscore.server (serving, built from the Go plugin's protobuf ResourceState) --
train/serve skew is the easiest way to silently break an RL policy, so this
is the single place either side is allowed to compute features.

A per-GPU list (milli_gpu_left_list) varies in length across nodes, so it is
summarized into fixed aggregate stats rather than fed in raw.
"""

from typing import List

from rlscore.frag import MILLI, NodeState, PodShape

FEATURE_DIM = 13

# Normalization constants. These are generous upper bounds (see
# data/csv/openb_node_list_all_node.csv: max cpu_milli is 128000, max gpu
# count per node is 8) rather than exact maxima, so the network sees roughly
# [0,1]-scaled inputs without needing to be refit if a slightly larger node
# shows up in a different trace.
MAX_MILLI_CPU = 256_000.0
MAX_GPU_PER_NODE = 8.0


def encode(node: NodeState, pod: PodShape) -> List[float]:
    gpu_lefts = node.milli_gpu_left_list
    fully_free = sum(1 for g in gpu_lefts if g >= MILLI)
    used = sum(1 for g in gpu_lefts if g <= 0)
    partial = len(gpu_lefts) - fully_free - used
    sum_left = sum(gpu_lefts)

    sufficient = [g for g in gpu_lefts if g >= pod.milli_gpu] if pod.milli_gpu > 0 else []
    best_fit_slack = (min(sufficient) - pod.milli_gpu) if sufficient else 0.0

    return [
        pod.milli_cpu / MAX_MILLI_CPU,
        pod.milli_gpu / MILLI,
        pod.gpu_number / MAX_GPU_PER_NODE,
        1.0 if pod.milli_gpu > 0 else 0.0,
        node.milli_cpu_left / MAX_MILLI_CPU,
        node.milli_cpu_capacity / MAX_MILLI_CPU,
        node.gpu_number / MAX_GPU_PER_NODE,
        fully_free / MAX_GPU_PER_NODE,
        used / MAX_GPU_PER_NODE,
        partial / MAX_GPU_PER_NODE,
        sum_left / (MAX_GPU_PER_NODE * MILLI),
        best_fit_slack / MILLI,
        (node.milli_cpu_left - pod.milli_cpu) / MAX_MILLI_CPU,
    ]
