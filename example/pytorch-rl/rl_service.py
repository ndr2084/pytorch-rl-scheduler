import argparse
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
##
app = Flask(__name__)

# Simple neural network policy
class Policy(nn.Module):
    def __init__(self, input_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output in scheduler score range 0-100
        return torch.sigmoid(self.net(x)) * 100.0

policy = Policy()

# --- Resource parsing helpers -------------------------------------------------
_CPU_M = 1000.0

_MEM_UNITS = {
    "Ki": 2 ** 10,
    "Mi": 2 ** 20,
    "Gi": 2 ** 30,
    "Ti": 2 ** 40,
}

def parse_cpu(s: str) -> float:
    if s.endswith("m"):
        return float(s[:-1]) / _CPU_M
    return float(s)

def parse_mem(s: str) -> float:
    for suf, mult in _MEM_UNITS.items():
        if s.endswith(suf):
            return float(s[:-len(suf)]) * mult
    return float(s)

# ----------------------------------------------------------------------------

def build_features(pod: dict, node: dict) -> torch.Tensor:
    cpu_req = 0.0
    mem_req = 0.0
    for c in pod.get("spec", {}).get("containers", []):
        req = c.get("resources", {}).get("requests", {})
        cpu_req += parse_cpu(req.get("cpu", "0"))
        mem_req += parse_mem(req.get("memory", "0"))
    alloc = node.get("status", {}).get("allocatable", {})
    cpu_alloc = parse_cpu(alloc.get("cpu", "0"))
    mem_alloc = parse_mem(alloc.get("memory", "0"))
    return torch.tensor([cpu_alloc, mem_alloc, cpu_req, mem_req], dtype=torch.float32)

@app.route('/score', methods=['POST'])
def score():
    payload = request.get_json(force=True)
    pod = payload.get('pod', {})
    nodes = payload.get('nodes', [])
    result = {}
    for node in nodes:
        name = node.get('metadata', {}).get('name', '')
        features = build_features(pod, node)
        with torch.no_grad():
            val = policy(features).item()
        result[name] = int(val)
    return jsonify({'scores': result})


def main():
    parser = argparse.ArgumentParser(description='RL scheduler service')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
