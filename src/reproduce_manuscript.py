"""Manuscript-synchronised reproduction entry point.

This wrapper extends the existing deterministic runner with the feature-only
MLP baseline used in the manuscript. All other behaviour is inherited from
``src/reproduce.py`` so the clean/artifact protocols, metrics, checkpointing,
and X-MoE options remain unchanged.
"""

from models import MLP
import reproduce as base


base.MODELS["mlp"] = MLP


if __name__ == "__main__":
    base.main()
