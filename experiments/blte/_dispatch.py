"""Labelled BLTE entry points; original scientific implementations stay intact."""
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
SEEDS = "100,101,102,103,104,105,106,107,108,109"

def launch(family):
    args = sys.argv[1:]
    scripts = {"clean": "src/runner.py", "sensitivity": "src/runner.py", "strict": "experiments/strict_inductive.py", "ablation": "experiments/ablation.py"}
    if args == ["--describe"]:
        print(f"dataset=blte; family={family}; implementation={scripts[family]}; default epochs=300; default seeds=100..109; default device=cpu")
        return
    def present(flag):
        return any(a == flag or a.startswith(flag + "=") for a in args)
    defaults = {"--epochs": "300", "--seeds": SEEDS, "--device": "cpu"}
    if family in ["clean", "sensitivity", "ablation"]:
        defaults["--protocol"] = "clean"
    if family == "sensitivity" and not present("--x-moe-normal-phishing-penalty") and not any(a in ["--help", "-h"] for a in args):
        raise SystemExit("Sensitivity requires an explicit --x-moe-normal-phishing-penalty value.")
    if family == "sensitivity":
        defaults["--model"] = "x_moe_gnn"
    defaults["--output-dir"] = str(ROOT / "results/new_runs/blte" / family)
    for flag, value in defaults.items():
        if not present(flag):
            args += [flag, value]
    target = ROOT / scripts[family]
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *args]
    runpy.run_path(str(target), run_name="__main__")
