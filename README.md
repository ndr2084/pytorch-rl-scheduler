# 🚀 Kubernetes Scheduler Simulator With a PyTorch RL Scheduler

## 🙏🏻 Acknowledge

This project seeks to extend the functionality of the simulator developed by [hkust-adsl](https://github.com/hkust-adsl/kubernetes-scheduler-simulator).

## 🚧 Environment Setup

1. Please ensure that Go is installed.

`go mod vendor` installs the dependencies required for the simulator.

```bash
$ go mod vendor
```

2. Please ensure the base Python dependencies are installed:

```bash
$ pip install -r requirements.txt
```

3. The RL scheduler itself (training + serving) has its own dependencies (PyTorch, gRPC, pandas) kept separate from the base `requirements.txt` above, since they're only needed if you're using `RLScore`:

```bash
$ rlscore/run.sh setup
```

## 🤔 How The Scheduler Was Implemented

After adding a new scheduler to `~./pkg/simulator/plugin`, we had to:
1. register the new policy under `func New(opts ...Option) (Interface, error)` in `~./pkg/simulator/simulator.go`
2. `make` to generate the compiled binary files in the `bin` directory.

```bash
$ make
```

You can follow steps 1 and 2 to implement your own scheduler as well. `RLScore` doesn't need a change to `pkg/simulator/utils.go`'s Go-side default plugin list — it's activated purely via a scheduler-config YAML's `pluginConfig`, the same way you'd pick between `FGDScore`/`BestFitScore`/etc. That's a deliberate choice: `RLScore` depends on an external process being reachable (see below), so it shouldn't be silently on by default.

## 🔥 How RLScore Talks To The Policy Server

The Go plugin (`pkg/simulator/plugin/rl_score.go`) and the Python policy server (`rlscore/server.py`) communicate over **gRPC**, not HTTP — the wire contract lives in `pkg/rlscore/proto/rlscore.proto`, and both sides are generated from that one file so they can't drift apart by hand-editing one side.

Start the policy server first (it needs to be running before you schedule anything against `RLScore`):

```bash
$ rlscore/run.sh serve
```

Then pass both a cluster configuration and a scheduler configuration to `simon apply`:

```bash
$ bin/simon apply --extended-resources "gpu" \
                  -f example/test-cluster-config.yaml \
                  -s example/rl-scheduler-config.yaml
```

`example/rl-scheduler-config.yaml` points `RLScore` at `localhost:50051` by default — see `rlscore/server.py --port` if you want it elsewhere.

### Training a policy

`rlscore/train.py` trains the network with PPO against a training environment (`rlscore/env.py`) that replays real `data/csv/openb_pod_list_*.csv` traces, using a reward ported directly from `pkg/utils/frag.go`'s fragmentation metric — the same quantity `FGDScorePlugin` minimizes, so a trained policy is directly comparable to that baseline.

```bash
$ rlscore/run.sh train
```

writes a checkpoint to `rlscore/checkpoint.pt`, which `rlscore/run.sh serve` loads automatically.

### Running your own scheduler script against RLScore

`scripts/generate_config_and_run.py` (the same tool used to generate configs for all the built-in baselines) knows about `RLScore` too — activate it the same way you'd activate any other policy, via `-RL <weight>`:

```bash
$ python3 scripts/generate_config_and_run.py -d <experiment-dir> -f <trace-folder> -RL 1000 -e -b
```

## 🚧 Topology Aware Functionality Added

`~./create_hierarchical_yamls.py` prepares hierarchical cluster YAMLs for the Kubernetes simulator
experiments.  It traverses the data directory, finds each workload directory
(matching ``openb_pod_list_*``), and produces modified copies of the node and
pod lists.  Node YAML documents are annotated with ``rack`` and ``server``
labels to mimic a rack/server hierarchy.

### Usage

Run the script from the repository root.  By default it targets the
``data`` directory relative to the current working directory and writes
modified YAMLs beside the originals with a ``-hier`` suffix.  You can
override the base directory, the number of servers per rack, and the
name of the custom scheduler via command‑line flags.

For example:

```bash

$ python create_hierarchical_yamls.py \
       --data-dir ./data \
       --servers-per-rack 4
```

This will produce files such as ``openb_node_list_gpu_node-hier.yaml``
and ``openb_pod_list_cpu050-hier.yaml`` in each ``openb_pod_list_*``
directory under ``./data``.

Racks and servers are assigned deterministically based on the order of
nodes in the node list: the first ``servers_per_rack`` nodes go into
rack 0 (servers 0, 1, …), the next ``servers_per_rack`` nodes go into
rack 1, and so on.  Adjust ``servers_per_rack`` for your own topology.

## 📜 History

This project's first attempt at an RL scheduler (`rl_scheduler_score.go`, communicating with a Flask HTTP service) is what `failure_log.txt` documents — it did not successfully schedule pods. It's been replaced by the gRPC/PyTorch implementation described above, which has been run end-to-end against the full real `openb_pod_list_default` trace (8,152 pods, 1,213 nodes) and benchmarked against the built-in baseline policies.
