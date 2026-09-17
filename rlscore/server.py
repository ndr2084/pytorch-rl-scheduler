"""gRPC server implementing the RLScorer service (pkg/rlscore/proto/rlscore.proto)
that pkg/simulator/plugin/rl_score.go dials into.

Usage (from the repo root, with rlscore/requirements.txt installed):

    python -m rlscore.server --checkpoint rlscore/checkpoint.pt --port 50051

If --checkpoint is omitted or the file doesn't exist, the server runs with a
freshly (randomly) initialized network so the wire contract is exercisable
end-to-end without a trained model -- scores will just be uninformative.
"""

import argparse
import logging
from concurrent import futures
from typing import List

import grpc
import torch

from rlscore.features import encode
from rlscore.frag import NodeState, PodShape
from rlscore.model import PolicyValueNet
from rlscore.proto_gen import rlscore_pb2 as pb2
from rlscore.proto_gen import rlscore_pb2_grpc as pb2_grpc

log = logging.getLogger("rlscore.server")


def _to_node_state(state: pb2.ResourceState) -> NodeState:
    return NodeState(
        name="",
        milli_cpu_capacity=state.milli_cpu_capacity,
        milli_cpu_left=state.milli_cpu_left,
        milli_gpu_left_list=list(state.milli_gpu_left_list),
        gpu_type="",  # not sent over the wire; see features.encode, which doesn't need it
    )


def _to_pod_shape(state: pb2.ResourceState) -> PodShape:
    return PodShape(milli_cpu=state.milli_cpu, milli_gpu=state.milli_gpu, gpu_number=state.gpu_number, gpu_type="")


GPU_ID_SEP = "-"  # must match gpushareutils.DevIdSep (pkg/type/open-gpu-share/utils/const.go)


def _pick_gpu_id(node: NodeState, pod: PodShape) -> str:
    """Mirrors the greedy GPU-slot choice rlscore.env._apply_placement makes
    during training, so the GPU actually bound at serving time matches what
    the policy was trained assuming would happen."""
    if pod.milli_gpu <= 0:
        return ""
    if pod.gpu_number > 1 or pod.milli_gpu >= 1000:
        chosen = [i for i, g in enumerate(node.milli_gpu_left_list) if g >= 1000][: pod.gpu_number]
        return GPU_ID_SEP.join(str(i) for i in chosen)
    sufficient = [(g, i) for i, g in enumerate(node.milli_gpu_left_list) if g >= pod.milli_gpu]
    if not sufficient:
        return ""
    _, gpu_idx = min(sufficient)
    return str(gpu_idx)


class RLScorerServicer(pb2_grpc.RLScorerServicer):
    def __init__(self, model: PolicyValueNet):
        self.model = model
        self.model.eval()

    def _score_one(self, req: pb2.ScoreRequest) -> pb2.ScoreResponse:
        node = _to_node_state(req.node)
        pod = _to_pod_shape(req.pod)
        features = torch.tensor([encode(node, pod)], dtype=torch.float32)
        with torch.no_grad():
            score = self.model.score(features)[0].item()
        return pb2.ScoreResponse(score=int(round(score)), gpu_id=_pick_gpu_id(node, pod))

    def ScorePlacement(self, request: pb2.ScoreRequest, context) -> pb2.ScoreResponse:
        return self._score_one(request)

    def ScoreBatch(self, request: pb2.BatchScoreRequest, context) -> pb2.BatchScoreResponse:
        if not request.requests:
            return pb2.BatchScoreResponse()

        nodes = [_to_node_state(r.node) for r in request.requests]
        pods = [_to_pod_shape(r.pod) for r in request.requests]
        features = torch.tensor([encode(n, p) for n, p in zip(nodes, pods)], dtype=torch.float32)
        with torch.no_grad():
            scores = self.model.score(features)

        responses: List[pb2.ScoreResponse] = []
        for node, pod, score in zip(nodes, pods, scores.tolist()):
            responses.append(pb2.ScoreResponse(score=int(round(score)), gpu_id=_pick_gpu_id(node, pod)))
        return pb2.BatchScoreResponse(responses=responses)


def load_model(checkpoint_path: str) -> PolicyValueNet:
    model = PolicyValueNet()
    if checkpoint_path:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            log.info("loaded checkpoint from %s", checkpoint_path)
            return model
        except FileNotFoundError:
            log.warning("checkpoint %s not found; serving a randomly-initialized (untrained) policy", checkpoint_path)
    else:
        log.warning("no --checkpoint given; serving a randomly-initialized (untrained) policy")
    return model


def serve(port: int, checkpoint_path: str, max_workers: int = 10) -> None:
    model = load_model(checkpoint_path)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    pb2_grpc.add_RLScorerServicer_to_server(RLScorerServicer(model), server)
    addr = f"[::]:{port}"
    server.add_insecure_port(addr)
    server.start()
    log.info("RLScorer serving on %s", addr)
    server.wait_for_termination()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--checkpoint", default="rlscore/checkpoint.pt")
    parser.add_argument("--max-workers", type=int, default=10)
    args = parser.parse_args()
    serve(args.port, args.checkpoint, args.max_workers)


if __name__ == "__main__":
    main()
