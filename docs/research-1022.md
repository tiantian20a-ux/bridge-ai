# Research summary (2026-10-22)

Why bridge: there is no permissively licensed (MIT/Apache) open engine that covers both bidding and card play with reproducible strength. Paying players are concentrated in English-speaking countries.

- Demand: BBO reports 11.6M monthly visits and over 1M monthly active players (https://news.bridgebase.com/about).
- Open competitors:
  - BEN (GPL-3.0): https://github.com/lorserker/ben
  - brl (Apache-2.0, bidding only, +1.24 IMP/board vs WBridge5): https://github.com/harukaki/brl
- Data: OpenSpiel 1M deals for supervised learning; Pgx 12.5M double-dummy (DDS) deals for RL (https://arxiv.org/html/2406.10306v1). All public, no scraping.
- Open questions:
  - Verify the DDS solver licence.
  - Get a student-visa work-rights opinion before accepting any income.
