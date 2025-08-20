# 🚀 Kubernetes Scheduler Simulator


## 🚧 Environment Setup

Please ensure that Go, pytorch, and flask are is installed.

`go mod vendor` installs the dependencies required for the simulator. 

```bash
$ go mod vendor
$ pip install torch
$ pip install flask
```

After adding a new scheduler to `~./pkg/simulator/plguin`, you must: 
1. add the scheduler to the appropriate plugin options under func GetAndSetSchedulerConfig in `~./pkg/simulator/utils.go`
2. register the new policy under func New(opts ...Option) (Interface, error) in `~./pkg/simulator/simulator.go`

`make` generates the compiled binary files in the `bin` directory.

```bash
$ make
```

## 🔥 Quickstart Example

The following example will schedule 6 pods to a cluster with 2 nodes, and the expected output will show the allocation ratio of each resource dimension (CPU, memory, GPU).
The default scheduling policy is fragmentation gradient descent (FGD).

```bash
$ bin/simon apply --extended-resources "gpu" \
                  -f example/test-cluster-config.yaml \
                  -s example/test-scheduler-config.yaml
```

## 🔮 Experiments on Production Traces

Install the required Python dependency environment.

```bash
$ pip install -r requirements.txt
```

1. Please refer to [README](data/README.md) under the `data` directory to prepare production traces.
2. Then refer to [README](experiments/README.md) under the `experiments` directory to reproduce the results reported in the paper.

## 📝 Paper

Please cite our paper if it is helpful to your research.

```
@inproceedings{FGD,
    title = {Beware of Fragmentation: Scheduling GPU-Sharing Workloads with Fragmentation Gradient Descent},
    author = {Qizhen Weng and Lingyun Yang and Yinghao Yu and Wei Wang and Xiaochuan Tang and Guodong Yang and Liping Zhang},
    booktitle = {2023 {USENIX} Annual Technical Conference},
    year = {2023},
    series = {{USENIX} {ATC} '23}
    url = {https://www.usenix.org/conference/atc23/presentation/weng},
    publisher = {{USENIX} Association},
}
```

## 🙏🏻 Acknowledge

Our simulator is developed based on [open-simulator](https://github.com/alibaba/open-simulator) by Alibaba, a simulator used for cluster capacity planning. 
This repository primarily evaluates the performance of different scheduling polices on production traces.
GPU-related plugin has been merged into the main branch of [open-simulator](https://github.com/alibaba/open-simulator).

## ⏳ TODO

- [ ] Add a minikube running example to demonstrate how the simulator schedules pods in a **real** Kubernetes cluster.
