"""GPU-share fragmentation metric, ported from pkg/utils/frag.go.

This is a direct port of NodeGpuShareFragAmount / GetNodePodFrag /
FragAmountSumExceptQ3 (see pkg/utils/frag.go), not an approximation: it is
the same "how much of this node's free GPU capacity is unusable by the
workload mix" score that FGDScorePlugin (pkg/simulator/plugin/fgd_score.go)
optimizes for in the Go scheduler. Using it as the RL reward means the
trained policy is directly comparable to the FGD baseline it is meant to
improve on, rather than optimizing an ad-hoc proxy.

The one deliberate simplification versus the Go implementation is the
*typical pod distribution* fed in (see trace.typical_pods): the Go simulator
derives it via a popularity-threshold walk over the live pod population
(simontype.DefaultTypicalPodPopularityThreshold); here it is the empirical
frequency of each distinct (cpu, gpu, gpu_number, gpu_type) request shape in
the training trace. The fragmentation math itself is unchanged.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

Q1_LACK_BOTH = "q1_lack_both"
Q2_LACK_GPU = "q2_lack_gpu"
Q3_SATISFIED = "q3_satisfied"
Q4_LACK_CPU = "q4_lack_cpu"
XL_SATISFIED = "xl_satisfied"
XR_LACK_CPU = "xr_lack_cpu"
NO_ACCESS = "no_access"

ALL_FRAG_TYPES = (Q1_LACK_BOTH, Q2_LACK_GPU, Q3_SATISFIED, Q4_LACK_CPU, XL_SATISFIED, XR_LACK_CPU, NO_ACCESS)

MILLI = 1000  # full-GPU capacity, matches gpushareutils.MILLI


@dataclass
class PodShape:
    """A (typical or real) pod's resource request. Mirrors simontype.PodResource."""

    milli_cpu: float
    milli_gpu: float  # per-GPU milli request, 0 for CPU-only pods
    gpu_number: int
    gpu_type: str = ""  # "" = no GPU type constraint; else "T1|T2" per pkg's convention


@dataclass
class NodeState:
    """A node's current free resources. Mirrors simontype.NodeResource."""

    name: str
    milli_cpu_capacity: float
    milli_cpu_left: float
    milli_gpu_left_list: List[float] = field(default_factory=list)
    gpu_type: str = ""

    @property
    def gpu_number(self) -> int:
        return len(self.milli_gpu_left_list)


def gpu_milli_left_total(node: NodeState) -> float:
    return sum(node.milli_gpu_left_list)


def is_node_accessible_to_pod(node_gpu_type: str, pod_gpu_type: str) -> bool:
    if not pod_gpu_type:
        return True
    if not node_gpu_type:
        return False  # pod wants a GPU type, node has no GPU at all
    wanted = [t for t in pod_gpu_type.split("|") if t]
    if not wanted:
        return True
    return node_gpu_type in wanted


def can_node_host_pod_on_gpu_memory(node: NodeState, pod: PodShape) -> bool:
    remaining = pod.gpu_number
    for g in node.milli_gpu_left_list:
        if g >= pod.milli_gpu:
            remaining -= 1
            if remaining <= 0:
                return True
    return False


def get_node_pod_frag(node: NodeState, pod: PodShape) -> str:
    if pod.milli_gpu == 0:
        return XL_SATISFIED if node.milli_cpu_left >= pod.milli_cpu else XR_LACK_CPU

    if not is_node_accessible_to_pod(node.gpu_type, pod.gpu_type):
        return NO_ACCESS

    if can_node_host_pod_on_gpu_memory(node, pod):
        return Q3_SATISFIED if node.milli_cpu_left >= pod.milli_cpu else Q4_LACK_CPU
    else:
        return Q2_LACK_GPU if node.milli_cpu_left >= pod.milli_cpu else Q1_LACK_BOTH


def get_gpu_frag_milli(node: NodeState, pod: PodShape) -> float:
    return sum(g for g in node.milli_gpu_left_list if g < pod.milli_gpu)


def node_gpu_share_frag_amount(node: NodeState, typical_pods: List[Tuple[PodShape, float]]) -> Dict[str, float]:
    """typical_pods: list of (shape, frequency in [0,1])."""
    amounts: Dict[str, float] = {t: 0.0 for t in ALL_FRAG_TYPES}
    gpu_total = gpu_milli_left_total(node)
    for pod, freq in typical_pods:
        if freq <= 0:
            continue
        frag_type = get_node_pod_frag(node, pod)
        if frag_type == Q3_SATISFIED:
            gpu_frag = get_gpu_frag_milli(node, pod)
            amounts[Q2_LACK_GPU] += freq * gpu_frag
            amounts[Q3_SATISFIED] += freq * (gpu_total - gpu_frag)
        else:
            amounts[frag_type] += freq * gpu_total
    return amounts


def frag_amount_sum_except_q3(amounts: Dict[str, float]) -> float:
    return sum(v for k, v in amounts.items() if k != Q3_SATISFIED)


def node_gpu_share_frag_score(node: NodeState, typical_pods: List[Tuple[PodShape, float]]) -> float:
    """Lower is better -- this is the quantity FGDScorePlugin minimizes."""
    return frag_amount_sum_except_q3(node_gpu_share_frag_amount(node, typical_pods))
