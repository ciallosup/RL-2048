"""CLI entry for rl2048-train."""

from rl2048.rl.config import load_config
from rl2048.rl.trainer import Trainer


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Train Masked Double DQN on 2048.")
    parser.add_argument("--config", type=Path, default=Path("configs/dqn_baseline.yaml"))
    parser.add_argument("--train-seed", type=int, default=None, help="Override config train_seed.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.train_seed is not None:
        config.train_seed = args.train_seed

    result = Trainer(config).train()
    print(f"Training finished: steps={result.env_steps}, episodes={result.episodes}")
    print(f"Checkpoint: {result.checkpoint_path}")


if __name__ == "__main__":
    main()
