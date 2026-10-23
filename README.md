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
- `src/bridge_ai/inference.py`: hand constraints from the auction (a few fixed rules until the bidding model exists)
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

Requires Python 3.11+.
