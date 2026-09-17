"""Trains the RL scheduling policy with PPO against SchedulingEnv, and
checkpoints it for rlscore.server to load.

Usage (from the repo root, with rlscore/requirements.txt installed):

    python -m rlscore.train \\
        --nodes data/csv/openb_node_list_gpu_node.csv \\
        --pods data/csv/openb_pod_list_gpushare60.csv \\
        --max-pods 2000 \\
        --iterations 200 \\
        --out rlscore/checkpoint.pt
"""

import argparse
import time

import torch

from rlscore.env import SchedulingEnv
from rlscore.model import PolicyValueNet
from rlscore.ppo import collect_rollout, ppo_update


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", required=True, help="node CSV, e.g. data/csv/openb_node_list_gpu_node.csv")
    p.add_argument("--pods", required=True, help="pod CSV, e.g. data/csv/openb_pod_list_gpushare60.csv")
    p.add_argument("--max-pods", type=int, default=None, help="cap the trace length (whole trace if omitted)")
    p.add_argument("--iterations", type=int, default=200, help="number of rollout/update cycles")
    p.add_argument("--steps-per-iteration", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4, help="PPO epochs per rollout")
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="rlscore/checkpoint.pt")
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    env = SchedulingEnv(args.nodes, args.pods, max_pods=args.max_pods)
    model = PolicyValueNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"[rlscore.train] {env.total_pods} pods, {len(env._node_template)} nodes, "
        f"{len(env.typical_pods)} typical-pod shapes"
    )

    obs = env.reset()
    start = time.time()
    for it in range(1, args.iterations + 1):
        buffer, last_value, episode_returns, obs = collect_rollout(env, model, args.steps_per_iteration, obs)
        stats = ppo_update(
            model,
            optimizer,
            buffer,
            last_value,
            gamma=args.gamma,
            lam=args.lam,
            clip_eps=args.clip_eps,
            epochs=args.epochs,
            minibatch_size=args.minibatch_size,
            entropy_coef=args.entropy_coef,
        )

        if it % args.log_every == 0:
            mean_ep_return = sum(episode_returns) / len(episode_returns) if episode_returns else float("nan")
            elapsed = time.time() - start
            print(
                f"[iter {it:4d}/{args.iterations}] "
                f"steps={it * args.steps_per_iteration:>7d} "
                f"episodes_completed={len(episode_returns):2d} "
                f"mean_ep_return={mean_ep_return:8.3f} "
                f"policy_loss={stats['policy_loss']:.4f} "
                f"value_loss={stats['value_loss']:.4f} "
                f"entropy={stats['entropy']:.4f} "
                f"placed={env.placed} dropped={env.dropped} "
                f"({elapsed:.1f}s elapsed)"
            )

    torch.save({"model_state_dict": model.state_dict()}, args.out)
    print(f"[rlscore.train] saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
