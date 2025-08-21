# 🚀 Kubernetes Scheduler Simulator With an MLP Scheduler

## 🙏🏻 Acknowledge

This project seeks to extend the functionality of the simulator developed by [hkust-adsl](https://github.com/hkust-adsl/kubernetes-scheduler-simulator). 


## 🚧 Environment Setup  

1.Please ensure that Go is installed.

`go mod vendor` installs the dependencies required for the simulator. 

```bash
$ go mod vendor
```
2. Please ensure the following are installed:  
`pip install torch` installs the dependencies required for the rl-based scheduler  
`pip install flask` installs the dependencies required to capture the pods from the cluster and assign scores via http requests  
`pip install numpy` torch has dependency on numpy
for convenience, we have added torch and flask in the requirements.txt file provided

```bash
$ pip install -r requirements.txt
```

## 🤔 How The Scheduler Was Implemented

After adding a new scheduler to `~./pkg/simulator/plguin`, we had to: 
1. add the scheduler to the appropriate plugin options under `func GetAndSetSchedulerConfig` in `~./pkg/simulator/utils.go`
2. register the new policy under `func New(opts ...Option) (Interface, error)` in `~./pkg/simulator/simulator.go`
3. `make` to generate the compiled binary files in the `bin` directory.
   
```bash
$ make
```
You can follow steps 1 through 3 to implement your own scheduler as well

## 🔥 Quickstart Example
### 🚧 Allow the MLP to receive http requests from the rl_scheduler_score.go plugin  

in directory `~./example/pytorch-rl`, run the scheduler so it can capture the pod data that will be sent from `~./pkg/simulator/plguin/rl_scheduler_score.go`

```bash
$ python rl_service.py
```

To create the simulation, both a cluster configuration file and a scheduler configuration file need to be passed as parameters to `simon apply` 

```bash
$ bin/simon apply --extended-resources "gpu" \
                  -f path/to/test-cluster-config.yaml \
                  -s path/to/test-scheduler-config.yaml
```


## 🚧  Topology Aware Functionality Added 

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

## ⏳ TODO

- [ ] Currently Pods are not being assigned to nodes. It needs to be determined whether:
  `~./example/pytorch-rl/rl_policy.py`  and/or
  `~./pkg/simulator/plugin/rl_scheduler_score.go` are to blame. [failure_log.txt](https://github.com/ndr2084/pytorch-rl-scheduler/blob/main/failure_log.txt) is a log of the failed integration of implementing our MLP based scheduler on the cluster. 
