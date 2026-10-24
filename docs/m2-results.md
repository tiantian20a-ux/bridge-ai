# M2 bidding training results (2026-10-24)

Our reproduction of brl's bidding pipeline, run with `scripts/autodl/m2_train.sh`
on an AutoDL RTX 4080 SUPER (32 GB) instance.

## Training

| stage | settings | result |
|---|---|---|
| SL | 400,000 iterations, batch 128, lr 1e-4, OpenSpiel WBridge5 data | test accuracy 0.9345, 69 minutes |
| RL | PPO + fictitious self-play from our SL model, 8192 envs × 32 steps, 10,000 updates (2.62 billion steps), brl's `model-pretrained-rl-with-fsp` settings | about 6.4 s per update, about 18 hours |

The RL run was interrupted at update 7430 when the instance stopped. It was
resumed on a second instance from the update-7400 checkpoint as run
`m2-fsp-from-00007400` for the remaining 2600 updates, with the 74 earlier
checkpoints seeded into the model zoo. The Adam state restarted at the resume.
The first ~50 updates of the resumed run were slower while the DDS hash tables
were rebuilt; after that the speed matched the original run.

## Evaluation

brl `eval.py`, duplicate, scored double dummy, 1000 boards per pair. Positive
means the first model wins.

| ours | brl's released model | IMPs/board |
|---|---|---|
| m2-sl | model-sl | −0.08 ± 0.15 |
| m2-fsp | model-sl | +1.58 ± 0.20 |
| m2-fsp | model-pretrained-rl-with-fsp | +0.67 ± 0.20 |

Our SL model is level with brl's SL model, and our RL model beats brl's
released FSP model by about three standard errors. Both trained models are
14.7 MB each:

- `m2-sl.pkl`  sha256 `cfeac50fda98385d92362c2a553e50b041de8741da64d30a9e8486332013bfd1`
- `m2-fsp.pkl` sha256 `8cfbfc23e065d9f902214f428862c050e1a20578693a0ed64118a1e53d3254bf`

They are not in this repository: the SL stage uses the OpenSpiel WBridge5
data, whose redistribution terms still need checking before the weights are
published.
