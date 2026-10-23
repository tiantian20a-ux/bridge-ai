"""Compare bidding models at duplicate, scored double dummy.

Each random board is bid twice: model A sits North-South at one table and
East-West at the other, against model B. Each contract is scored with the
double-dummy number of tricks, and the difference between the tables is
converted to IMPs. This is how brl measures its models against each other
(its paper uses WBridge5 as the opponent instead).

    python scripts/eval_bidding.py data/brl_models/model-pretrained-rl-with-pfsp.pkl \\
        data/brl_models/model-sl.pkl --boards 1000
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from pathlib import Path

from bridge_ai import dds
from bridge_ai.bidding import NeuralBidder, PolicyNet, bid_out
from bridge_ai.deal import Deal, Seat, Vulnerability, dealer_for_board
from bridge_ai.scoring import imps, score_ns


def table_score(deal: Deal, table: dds.DDTable, ns: NeuralBidder, ew: NeuralBidder) -> tuple[int, str]:
    bidders = {s: ns if s % 2 == 0 else ew for s in Seat}
    auction = bid_out(deal.hands, deal.dealer, deal.vulnerability, bidders)
    contract = auction.contract()
    if contract is None:
        return 0, "passed out"
    tricks = table[contract.denomination, contract.declarer]
    return score_ns(contract, tricks, deal.vulnerability), str(contract)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_a", type=Path)
    ap.add_argument("model_b", type=Path)
    ap.add_argument("--boards", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", type=int, default=0, help="print this many boards where the tables differ")
    args = ap.parse_args()

    a, b = NeuralBidder(PolicyNet.load(args.model_a)), NeuralBidder(PolicyNet.load(args.model_b))
    rng = random.Random(args.seed)
    deals = []
    for n in range(1, args.boards + 1):
        d = Deal.random(rng)
        deals.append(Deal(d.hands, dealer_for_board(n), Vulnerability.for_board(n)))
    tables = dds.dd_tables(deals)

    results, shown = [], 0
    for deal, table in zip(deals, tables):
        s1, c1 = table_score(deal, table, a, b)  # A sits NS
        s2, c2 = table_score(deal, table, b, a)  # A sits EW
        results.append(imps(s1 - s2))
        if c1 != c2 and shown < args.show:
            shown += 1
            print(f"{deal.to_pbn()}  dealer {deal.dealer.symbol} vul {deal.vulnerability.value}: "
                  f"A-NS {c1} ({s1:+}), A-EW {c2} ({s2:+}) -> {results[-1]:+} IMPs")

    mean = statistics.mean(results)
    err = statistics.stdev(results) / math.sqrt(len(results)) if len(results) > 1 else float("nan")
    same = sum(r == 0 for r in results)
    print(f"{args.model_a.stem} vs {args.model_b.stem}: {mean:+.3f} IMPs/board (±{err:.3f}), "
          f"{len(results)} boards, {same} with no swing")


if __name__ == "__main__":
    main()
