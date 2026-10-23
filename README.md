# bridge-ai

Open-source contract bridge AI covering both bidding and card play, released under a permissive licence.

Status: planning. See `docs/research-1022.md` for the market and competitor survey.

## Layout

- `src/bridge_ai/cards.py`: cards (0–51) and hands (52-bit masks), PBN hand I/O, HCP and shape
- `src/bridge_ai/deal.py`: seats, vulnerability, deals, random dealing, PBN deal I/O
- `src/bridge_ai/auction.py`: calls (38-way index), auction legality, contract and declarer
- `src/bridge_ai/state.py`: game state from auction through card play, tricks
- `src/bridge_ai/scoring.py`: duplicate scoring and IMP conversion
- `src/bridge_ai/dds.py`: double-dummy solver (ctypes binding to DDS 2.9), single and batched
- `src/bridge_ai/sampling.py`: layouts of the unseen cards consistent with one seat's view of the play
- `src/bridge_ai/inference.py`: hand constraints from the auction (a few fixed rules, the fallback without a bidding model)
- `src/bridge_ai/bidding.py`: brl bidding networks run with numpy: a bidder, and weights that tell PIMC how well a layout explains the auction
- `src/bridge_ai/play.py`: card-play agents (PIMC, double dummy, simple rules, random) and a play-out loop

## Development

```
bash scripts/build_dds.sh   # builds DDS into third_party/dds/ (needs git, make, a C++ compiler)
pip install -e .[dev]
pytest
```

To check the DDS binding against an independent build (endplay):

```
pip install -e .[verify]
python scripts/verify_dds.py --deals 1000
```

To measure the card-play agents against double-dummy play:

```
python scripts/eval_play.py --deals 200 --samples 64
```

Neural bidding uses brl's released models (Apache-2.0, about 15 MB each, not committed):

```
pip install -e .[bidding]
bash scripts/fetch_brl_models.sh
python scripts/eval_bidding.py data/brl_models/model-pretrained-rl-with-pfsp.pkl data/brl_models/model-sl.pkl --boards 1000
python scripts/eval_inference.py data/brl_models/model-pretrained-rl-with-pfsp.pkl --boards 200
```

`eval_bidding.py` plays two models against each other at duplicate, scored double dummy;
`eval_inference.py` measures how much reading the auction with the network helps PIMC.

Requires Python 3.11+.
