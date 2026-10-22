# bridge-ai

Open-source contract bridge AI covering both bidding and card play, released under a permissive licence.

Status: planning. See `docs/research-1022.md` for the market and competitor survey.

## Layout

- `src/bridge_ai/cards.py`: cards (0–51) and hands (52-bit masks), PBN hand I/O, HCP and shape
- `src/bridge_ai/deal.py`: seats, vulnerability, deals, random dealing, PBN deal I/O
- `src/bridge_ai/auction.py`: calls (38-way index), auction legality, contract and declarer
- `src/bridge_ai/state.py`: game state from auction through card play, tricks
- `src/bridge_ai/scoring.py`: duplicate scoring and IMP conversion
- `src/bridge_ai/dds.py`: double-dummy solver (ctypes binding to DDS 2.9)

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

Requires Python 3.11+.
