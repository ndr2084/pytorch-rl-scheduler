"""Loads the openb_* CSV traces (see data/README.md) into the NodeState /
PodShape dataclasses used by rlscore.frag and rlscore.env, and derives an
empirical "typical pod" distribution used as fragmentation-reward weights.
"""

from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

from rlscore.frag import MILLI, NodeState, PodShape


@dataclass
class ArrivingPod:
    shape: PodShape
    name: str
    creation_time: float
    deletion_time: float  # inf if unknown/never observed departing


def load_nodes(node_csv_path: str) -> List[NodeState]:
    df = pd.read_csv(node_csv_path)
    nodes = []
    for _, row in df.iterrows():
        gpu_count = int(row["gpu"]) if pd.notna(row["gpu"]) else 0
        gpu_type = str(row["model"]) if gpu_count > 0 and pd.notna(row["model"]) else ""
        nodes.append(
            NodeState(
                name=str(row["sn"]),
                milli_cpu_capacity=float(row["cpu_milli"]),
                milli_cpu_left=float(row["cpu_milli"]),
                milli_gpu_left_list=[float(MILLI)] * gpu_count,
                gpu_type=gpu_type,
            )
        )
    return nodes


def _row_to_pod_shape(row) -> PodShape:
    gpu_number = int(row["num_gpu"]) if pd.notna(row["num_gpu"]) else 0
    if gpu_number <= 0:
        milli_gpu = 0.0
    elif pd.notna(row.get("gpu_milli")) and row["gpu_milli"] > 0:
        milli_gpu = float(row["gpu_milli"])
    else:
        milli_gpu = float(MILLI)
    gpu_type = str(row["gpu_spec"]) if pd.notna(row.get("gpu_spec")) else ""
    return PodShape(
        milli_cpu=float(row["cpu_milli"]),
        milli_gpu=milli_gpu,
        gpu_number=gpu_number,
        gpu_type=gpu_type,
    )


def load_pods(pod_csv_path: str, max_pods: int = None) -> List[ArrivingPod]:
    df = pd.read_csv(pod_csv_path)
    df = df.dropna(subset=["creation_time"]).sort_values("creation_time")
    if max_pods is not None:
        df = df.iloc[:max_pods]

    pods = []
    for _, row in df.iterrows():
        deletion_time = row["deletion_time"] if pd.notna(row.get("deletion_time")) else float("inf")
        pods.append(
            ArrivingPod(
                shape=_row_to_pod_shape(row),
                name=str(row["name"]),
                creation_time=float(row["creation_time"]),
                deletion_time=float(deletion_time),
            )
        )
    return pods


def typical_pod_distribution(pods: List[ArrivingPod], min_frequency: float = 0.01) -> List[Tuple[PodShape, float]]:
    """Empirical frequency of each distinct request shape in the trace.

    This stands in for the popularity-threshold walk
    (simontype.DefaultTypicalPodPopularityThreshold) the Go simulator uses to
    build simontype.TargetPodList -- see the module docstring in frag.py.
    """
    counts = Counter((p.shape.milli_cpu, p.shape.milli_gpu, p.shape.gpu_number, p.shape.gpu_type) for p in pods)
    total = sum(counts.values())
    typical = []
    for (milli_cpu, milli_gpu, gpu_number, gpu_type), count in counts.items():
        freq = count / total
        if freq >= min_frequency:
            typical.append((PodShape(milli_cpu, milli_gpu, gpu_number, gpu_type), freq))
    if not typical:  # trace too small/uniform for the threshold; fall back to everything
        typical = [(PodShape(c, g, n, t), cnt / total) for (c, g, n, t), cnt in counts.items()]
    return typical
