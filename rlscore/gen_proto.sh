#!/usr/bin/env bash
# Regenerates rlscore/proto_gen/{rlscore_pb2.py,rlscore_pb2_grpc.py} from the
# same pkg/rlscore/proto/rlscore.proto the Go plugin (pkg/simulator/plugin/rl_score.go)
# uses, so the Python policy server and Go client always share one wire contract.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m grpc_tools.protoc \
  -I pkg/rlscore/proto \
  --python_out=rlscore/proto_gen \
  --grpc_python_out=rlscore/proto_gen \
  pkg/rlscore/proto/rlscore.proto

# grpc_tools emits a plain "import rlscore_pb2 as rlscore__pb2" which only
# resolves if proto_gen/ is on sys.path directly; rewrite it to a package-
# relative import so `python -m rlscore.server` works from the repo root.
sed -i 's/^import rlscore_pb2 as rlscore__pb2$/from . import rlscore_pb2 as rlscore__pb2/' \
  rlscore/proto_gen/rlscore_pb2_grpc.py

echo "Generated rlscore/proto_gen/rlscore_pb2.py and rlscore_pb2_grpc.py"
