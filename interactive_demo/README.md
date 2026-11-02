# BLTE five-model prediction demo

This demo loads MLP, GCN, GAT, GraphSAGE and X-MoE seed-100 checkpoints.
The research result charts show ten-seed means and sample standard deviations;
individual-address predictions come from seed 100 only. No training occurs in the app.

From the repository root, use an environment compatible with demo/requirements.txt:

```powershell
python -m streamlit run .\demo\streamlit_app.py
```

To validate checkpoint inference against every saved seed-100 test prediction:

```powershell
python .\interactive_demo\engine.py
```

Validation requires class agreement on all 10,955 test addresses for every model,
and probability agreement within atol=1e-5, rtol=1e-4. `manifest.json` checks the
packaged inputs, model code and checkpoint assets before inference.
`validation_reference.json` records validation in the integration environment;
engine.py writes a separate validation_local.json when run subsequently.

The 23 features, class order and original training-only scaler are preserved.
MLP ignores graph edges. The graph models retain full released transductive context.
Only held-out benchmark test addresses are accepted; the app does not retrieve
live blockchain data or assess arbitrary new addresses. Probabilities are model
outputs, not calibrated guarantees of safety. Routing diagnostics are not causal explanations.

The MLP checkpoint was regenerated at seed 100 without replacing the original
ten-seed result series. Integration provenance is in `mlp_integration_provenance.json`.
The other four checkpoints and their saved prediction references are unchanged.

Dataset attribution:
https://github.com/salam-ammari/Graph-Neural-Network-Based-Fraudulent-Address-Classification-in-Ethereum-Network

The package contains derived benchmark inputs. Dataset ownership and upstream
licensing terms remain with the original providers. Other datasets are excluded.

## Five-model protocol results

The Streamlit Protocol Comparison page reports all five models under clean and strict, with ten-seed means/SD and explicit Holm families. Try a Prediction continues to use clean seed-100 checkpoints. Strict aggregate results do not replace these checkpoints. See `results/paper/` and `docs/RELEASE_ALIGNMENT.md`.
