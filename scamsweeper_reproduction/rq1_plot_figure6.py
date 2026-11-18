"""Recreate ScamSweeper Figure 6 from full RQ1 degree histograms."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib")
)

import matplotlib.pyplot as plt


LABELS = {
    "normal": "Normal",
    "phish": "Phishing",
    "scams": "Web3 Scams",
}


def read_histogram(path: Path) -> tuple[list[int], list[int], list[int]]:
    degrees: list[int] = []
    incoming: list[int] = []
    outgoing: list[int] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            degree = int(row["degree"])
            if degree <= 0:
                continue
            in_frequency = int(row["in_frequency"])
            out_frequency = int(row["out_frequency"])
            if in_frequency or out_frequency:
                degrees.append(degree)
                incoming.append(in_frequency)
                outgoing.append(out_frequency)
    return degrees, incoming, outgoing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for axis, graph in zip(axes, ("normal", "phish", "scams"), strict=True):
        degrees, incoming, outgoing = read_histogram(
            args.input_dir / f"{graph}_degree_frequency.csv"
        )
        in_x = [degree for degree, frequency in zip(degrees, incoming, strict=True) if frequency]
        in_y = [frequency for frequency in incoming if frequency]
        out_x = [degree for degree, frequency in zip(degrees, outgoing, strict=True) if frequency]
        out_y = [frequency for frequency in outgoing if frequency]
        axis.loglog(in_x, in_y, color="#1976d2", linewidth=1.0, label="in-degree")
        axis.loglog(out_x, out_y, color="#ff7f0e", linewidth=1.0, linestyle="--", label="out-degree")
        axis.set_title(LABELS[graph])
        axis.set_xlabel("Degree")
        axis.grid(True, which="both", linewidth=0.3, alpha=0.25)
        axis.legend(frameon=True, fontsize=8)
    axes[0].set_ylabel("Frequency")
    figure.suptitle("RQ1 In-degree and Out-degree Frequency Distributions", fontsize=12)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()

