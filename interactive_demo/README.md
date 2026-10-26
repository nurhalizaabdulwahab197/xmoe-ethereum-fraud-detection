# X-MoE interactive prediction demo

This package adds **Try a Prediction** to the existing Streamlit research website.
It runs the four supplied seed-100 checkpoints; it does not use prediction CSVs
as a substitute for inference. Results are cached after each model's first
full-graph forward pass. No training is performed.

## Local installation (PowerShell)

Extract the ZIP into `C:\Users\CS-NUR.HALIZA\Documents\ChatGPT\fraud`.
The resulting folder should be `XMoE_Interactive_Demo`.

```powershell
$fraudRoot = "C:\Users\CS-NUR.HALIZA\Documents\ChatGPT\fraud"
$demoPython = "$fraudRoot\xmoe-ethereum-fraud-detection\.venv-cpu\Scripts\python.exe"
& $demoPython "$fraudRoot\XMoE_Interactive_Demo\install_into_website.py" --website "$fraudRoot\xmoe-website-icon"
```

Refresh the existing localhost:8502 app. If it is stopped:

```powershell
cd "$fraudRoot\xmoe-website-icon"
& $demoPython -m streamlit run .\demo\streamlit_app.py --server.port=8502
```

The installer backs up the current app and any earlier interactive_demo folder
in the Windows TEMP directory. It adds a menu option and handler and copies
inference files into `interactive_demo`. Existing architecture and results pages
and the training repository are preserved. It does not change dependencies,
commit, push or publish. Do not run the training reproduction script to launch
the demo.

Alternatively test independently without editing the website:

```powershell
& $demoPython -m streamlit run "$fraudRoot\XMoE_Interactive_Demo\app.py" --server.port=8503
```

Use the Streamlit launcher, not direct python app.py, for a browser session.

## What the page supports

- Ethereum address input, restricted to the 10,955 held-out seed-100 test addresses.
- Nine labelled examples (three for each class).
- Model selection and per-class probability comparison.
- Dataset ground truth reveal, raw features and graph connectivity.
- X-MoE expert probabilities and per-class routing diagnostics.
- Batch comparison of up to 50 addresses, pasted or uploaded as CSV (`Address` column).
- CSV exports. Unknown addresses receive no prediction.

## Scientific scope

This is a frozen **transductive BLTE benchmark demonstration**, not a live
blockchain screening service. It does not call an explorer/API or build features
for newly entered arbitrary addresses. Labels are Normal, Phishing and Scamming;
Normal does not establish that an address is currently safe. Percentages are
uncalibrated model probabilities, not individual-address accuracy.

The released graph is label-correlated. Thus held-out-node prediction parity is
not evidence of generalisation to newly collected/future/cross-chain fraud.
Per-class routing weights are intrinsic model diagnostics, not causal
attributions, and fusion operates on logits rather than a weighted average of
the probabilities shown in the expert table.

## Preprocessing and validation

The source data contain 73,033 addresses, 23 features and 6,588 released edges.
The original clean-protocol split is regenerated with seed 100 (70/15/15;
stratified scikit-learn train_test_split). StandardScaler is fitted on 51,123
training rows only. The derived NPZ preserves the exact feature order, raw
features, fitted scaler statistics, addresses, labels, edges and test mask.
Inference never refits a scaler and never receives ground-truth labels as model
inputs. Full-graph context is retained, matching the original experiment.

The legacy checkpoint prefix `v8_expert.` is mapped to
`conflict_aware_expert.` at load time. All weights are loaded with strict=True;
checkpoint class and feature order and clean protocol/seed are checked.
Class bias values come from checkpoint metrics. There is no silently ignored
missing-weight fallback.

All four models matched every saved test prediction, with maximum absolute
probability discrepancies below 0.000001. See validation_reference.json for
actual results and environment. This reference was obtained on Linux CPU;
run the following in the user's Windows environment to check local parity:

```powershell
& $demoPython "$fraudRoot\XMoE_Interactive_Demo\engine.py"
```

The command writes validation_local.json and exits nonzero on a failed check.
The tolerance is absolute 1e-5, relative 1e-4, with exact predicted-class
agreement required for all test nodes. Input/model/source hashes in manifest.json
are checked before inference.

## Live deployment later

The existing website's demo/requirements.txt only supported result displays.
Live inference also requires torch, torch-geometric, NumPy and the packaged
inputs/checkpoints. requirements-inference.txt is supplied for a new CPU
environment; validate hosting Python/wheel compatibility and cold-start memory
before updating the deployment requirements. Do not blindly install all packages
from the original research environment. Local integration intentionally leaves
live deployment requirements unchanged.

## Dataset provenance

Original processed artifacts were provided by the user from:
https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network

Dataset ownership and upstream usage/licensing terms remain with the original
authors. This package does not claim ownership of those artifacts. Original input
SHA-256 values are retained in manifest.json. The ZIP contains derived benchmark
inputs rather than another copy of the original graph pickle/CSV.
