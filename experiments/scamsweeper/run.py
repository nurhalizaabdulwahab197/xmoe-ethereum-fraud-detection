"""Dataset-labelled launcher for the byte-preserved Sweeper/RQ1 CUDA suite."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / "source"

def main():
    args = sys.argv[1:]
    if args == ["--describe"]:
        print("dataset=scamsweeper; internal key=rq1; clean=50 strict=50 ablation=80 sensitivity=50; cuda=230; seeds=100..109; epochs=300")
        return
    if "--include-gpu" in args:
        raise SystemExit("The locked matrix is already all-CUDA; --include-gpu is not supported by this entry point.")
    if not any(a == "--output" or a.startswith("--output=") for a in args) and not any(a in ["--help", "-h"] for a in args):
        raise SystemExit("Supply --output with a NEW experiment directory; locked evidence must not be overwritten.")
    # Resolve path arguments relative to the caller, before using the archived cwd.
    for flag in ["--graph", "--output"]:
        for i, arg in enumerate(args):
            if arg == flag and i+1 < len(args):
                args[i+1] = str(Path(args[i+1]).resolve())
            elif arg.startswith(flag + "="):
                args[i] = flag + "=" + str(Path(arg.split("=",1)[1]).resolve())
    command = [sys.executable, str(SOURCE / "experiments/rq1_parity/run.py"), *args]
    raise SystemExit(subprocess.call(command, cwd=SOURCE))

if __name__ == "__main__":
    main()
