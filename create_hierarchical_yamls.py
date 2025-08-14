"""
create_hierarchical_yamls.py
================================

This script prepares hierarchical cluster YAMLs for the Kubernetes simulator
experiments.  It traverses the data directory, finds each workload directory
(matching ``openb_pod_list_*``), and produces modified copies of the node and
pod lists.  Node YAML documents are annotated with ``rack`` and ``server``
labels to mimic a rack/server hierarchy, and pod YAML documents are updated
to specify the custom scheduler name required by an external RL scheduler.

Usage
-----

Run the script from the repository root.  By default it targets the
``data`` directory relative to the current working directory and writes
modified YAMLs beside the originals with a ``-hier`` suffix.  You can
override the base directory, the number of servers per rack, and the
name of the custom scheduler via command‑line flags.

For example:

.. code-block:: bash

   python3 create_hierarchical_yamls.py \
       --data-dir ./data \
       --servers-per-rack 4 \
       --scheduler-name rl-scheduler

This will produce files such as ``openb_node_list_gpu_node-hier.yaml``
and ``openb_pod_list_cpu050-hier.yaml`` in each ``openb_pod_list_*``
directory under ``./data``.

Racks and servers are assigned deterministically based on the order of
nodes in the node list: the first ``servers_per_rack`` nodes go into
rack 0 (servers 0, 1, …), the next ``servers_per_rack`` nodes go into
rack 1, and so on.  Adjust ``servers_per_rack`` for your own topology.

.. note::

   Only the scheduler name is applied to pods; no node selectors or
   affinities are added.  The external RL scheduler is responsible for
   deciding which node each pod should bind to.  If you wish to restrict
   certain pods to specific racks or servers, you can further modify the
   generated YAMLs to include ``nodeSelector`` fields, but doing so
   bypasses the RL scheduler’s decision making.

"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import yaml

def annotate_nodes_with_hierarchy(nodes: List[dict], servers_per_rack: int) -> List[dict]:
    """Add ``rack`` and ``server`` labels to each node.

    The rack index and server index are derived from the position of the node
    within the list.  For example, with ``servers_per_rack=4``, nodes 0–3
    belong to ``rack-0`` and have server labels ``srv-0`` … ``srv-3``;
    nodes 4–7 belong to ``rack-1`` and so on.

    Parameters
    ----------
    nodes:
        A list of parsed Node objects (as dictionaries) from a multi‑doc YAML.
    servers_per_rack:
        The number of servers to assign to each rack.

    Returns
    -------
    List[dict]
        A new list of node dictionaries with updated labels.
    """
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        rack_idx = idx // servers_per_rack
        server_idx = idx % servers_per_rack
        metadata = node.setdefault("metadata", {})
        labels = metadata.setdefault("labels", {})
        # Do not clobber existing labels if present.
        if "rack" not in labels:
            labels["rack"] = f"rack-{rack_idx}"
        if "server" not in labels:
            labels["server"] = f"srv-{server_idx}"
    return nodes


def update_pods_with_scheduler(pods: List[dict], scheduler_name: str) -> List[dict]:
    """Add ``schedulerName`` to each pod spec if not already present.

    Parameters
    ----------
    pods:
        A list of parsed Pod objects (as dictionaries) from a multi‑doc YAML.
    scheduler_name:
        The name of the custom scheduler (e.g. ``rl-scheduler``).

    Returns
    -------
    List[dict]
        A new list of pod dictionaries with ``spec.schedulerName`` set.
    """
    for pod in pods:
        if not isinstance(pod, dict):
            continue
        spec = pod.setdefault("spec", {})
        # Only set schedulerName if not already defined to avoid overriding
        # existing configuration.
        spec.setdefault("schedulerName", scheduler_name)
    return pods


def process_directory(dir_path: Path, servers_per_rack: int, scheduler_name: str) -> None:
    """Process a single workload directory.

    This function reads the node and pod YAML files in ``dir_path``, annotates
    them, and writes new files with a ``-hier`` suffix.  It emits no output
    if the expected files are not found.
    """
    node_yaml = None
    pod_yaml = None
    for item in dir_path.iterdir():
        if item.name.startswith("openb_node_list") and item.suffix == ".yaml":
            node_yaml = item
        elif item.name.startswith("openb_pod_list") and item.suffix == ".yaml":
            pod_yaml = item

    if node_yaml is None or pod_yaml is None:
        # Nothing to do in this directory
        return

    # Read multi‑doc YAML for nodes
    with open(node_yaml, "r") as f:
        nodes = list(yaml.safe_load_all(f))
    nodes = annotate_nodes_with_hierarchy(nodes, servers_per_rack)
    # Write annotated nodes YAML
    node_out = node_yaml.with_name(node_yaml.stem + "-hier" + node_yaml.suffix)
    with open(node_out, "w") as f:
        yaml.safe_dump_all(nodes, f)

    # Read multi‑doc YAML for pods
    with open(pod_yaml, "r") as f:
        pods = list(yaml.safe_load_all(f))
    pods = update_pods_with_scheduler(pods, scheduler_name)
    # Write updated pods YAML
    pod_out = pod_yaml.with_name(pod_yaml.stem + "-hier" + pod_yaml.suffix)
    with open(pod_out, "w") as f:
        yaml.safe_dump_all(pods, f)

    print(f"Processed {dir_path.name}: wrote {node_out.name}, {pod_out.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hierarchical YAMLs for workload directories")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base directory containing openb_pod_list_* directories (default: ./data)",
    )
    parser.add_argument(
        "--servers-per-rack",
        type=int,
        default=4,
        help="Number of servers per rack when assigning rack/server labels",
    )
    parser.add_argument(
        "--scheduler-name",
        type=str,
        default="rl-scheduler",
        help="Name of the custom scheduler to set on pods (default: rl-scheduler)",
    )
    args = parser.parse_args()

    base_dir = args.data_dir
    if not base_dir.exists():
        print(f"Data directory {base_dir} does not exist")
        return

    for child in base_dir.iterdir():
        if child.is_dir() and child.name.startswith("openb_pod_list"):
            process_directory(child, args.servers_per_rack, args.scheduler_name)

if __name__ == "__main__":
    main()