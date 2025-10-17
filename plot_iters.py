import argparse
import re
from pathlib import Path
from typing import List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "matplotlib is required for plotting. Install it with 'pip install matplotlib'."
    ) from exc


def parse_log(path: Path) -> Tuple[List[int], List[float]]:
    iterations: List[int] = []
    speeds: List[float] = []
    for raw_line in path.read_text().splitlines():
        iter_match = re.search(r'(\d+)\s*/\s*\d+', raw_line)
        if not iter_match:
            continue
        speed_match = re.search(r'([\d.]+)\s*it/s', raw_line)
        if speed_match:
            speed = float(speed_match.group(1))
        else:
            inv_match = re.search(r'([\d.]+)\s*s/it', raw_line)
            if not inv_match:
                continue
            seconds = float(inv_match.group(1))
            if seconds == 0:
                continue
            speed = 1.0 / seconds
        iterations.append(int(iter_match.group(1)))
        speeds.append(speed)
    return iterations, speeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot iterations-per-second over iteration for two logs."
    )
    parser.add_argument(
        "llama_log",
        type=Path,
        nargs="?",
        default=Path("llama.txt"),
        help="Path to the Llama log file (default: llama.txt).",
    )
    parser.add_argument(
        "qwen_log",
        type=Path,
        nargs="?",
        default=Path("qwen.txt"),
        help="Path to the Qwen log file (default: qwen.txt).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If set, save the figure to this path instead of showing it.",
    )
    parser.add_argument(
        "--llama-label",
        default="Llama 3.1 8B",
        help="Legend label for the Llama series.",
    )
    parser.add_argument(
        "--qwen-label",
        default="Qwen 3 8B",
        help="Legend label for the Qwen series.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    llama_iters, llama_speeds = parse_log(args.llama_log)
    qwen_iters, qwen_speeds = parse_log(args.qwen_log)

    if not llama_iters:
        raise ValueError(f"No iteration data extracted from {args.llama_log}")
    if not qwen_iters:
        raise ValueError(f"No iteration data extracted from {args.qwen_log}")

    plt.figure(figsize=(10, 6))
    plt.plot(llama_iters, llama_speeds, label=args.llama_label)
    plt.plot(qwen_iters, qwen_speeds, label=args.qwen_label)
    plt.xlabel("Iteration")
    plt.ylabel("Iterations per second (it/s)")
    plt.title("Iteration Throughput Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    if args.save:
        plt.tight_layout()
        plt.savefig(args.save, dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
