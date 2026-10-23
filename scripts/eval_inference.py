"""Measure how much reading the auction helps PIMC card play.

A bidding network bids random boards at all four seats. On each board
PIMC then plays one side against double-dummy opponents, as in
eval_play.py, reading the auction three ways:

- none:   samples ignore the auction
- rules:  the fixed natural-system rules in inference.py
- neural: samples are weighed by how likely the network is to have made
          the hidden seats' calls with them (bidding.AuctionModel)

    python scripts/eval_inference.py data/brl_models/model-pretrained-rl-with-pfsp.pkl --boards 100
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

from bridge_ai import dds
from bridge_ai.bidding import AuctionModel, NeuralBidder, PolicyNet, bid_out
from bridge_ai.deal import Deal, Seat, Vulnerability, dealer_for_board
from bridge_ai.play import DoubleDummyPlayer, PIMCPlayer, play_out
from bridge_ai.state import GameState


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("--boards", type=int, default=100)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--oversample", type=int, default=8)
    ap.add_argument("--modes", nargs="+", default=["none", "rules", "neural"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    net = PolicyNet.load(args.model)
    bidder = NeuralBidder(net)
    rng = random.Random(args.seed)
    boards = []
    n = 0
    while len(boards) < args.boards:
        n += 1
        d = Deal.random(rng)
        deal = Deal(d.hands, dealer_for_board(n), Vulnerability.for_board(n))
        auction = bid_out(deal.hands, deal.dealer, deal.vulnerability, {s: bidder for s in Seat})
        if auction.contract() is not None:
            boards.append((deal, auction))
    tables = dds.dd_tables(d for d, _ in boards)

    dd = DoubleDummyPlayer()
    print(f"{args.boards} boards bid by {args.model.stem}, PIMC {args.samples} samples")
    print(f"{'mode':8} {'role':9} {'tricks lost':>12} {'made':>8} {'s/board':>8}")
    for mode in args.modes:
        for role in ("declarer", "defence"):
            lost, hit, t0 = [], 0, time.perf_counter()
            for (deal, auction), table in zip(boards, tables):
                agent = PIMCPlayer(args.samples, use_auction=mode == "rules", rng=random.Random(args.seed))
                if mode == "neural":
                    agent.auction_model = AuctionModel(net, deal.vulnerability)
                    agent.oversample = args.oversample
                g = GameState(deal)
                for call in auction.calls:
                    g.make_call(call)
                c = auction.contract()
                dd_tricks = table[c.denomination, c.declarer]
                ours = {c.declarer, c.declarer.partner}
                if role == "defence":
                    ours = {c.declarer.next(), c.declarer.next(3)}
                tricks = play_out(g, {s: agent if s in ours else dd for s in Seat})
                if role == "declarer":
                    lost.append(dd_tricks - tricks)
                    hit += tricks >= c.level + 6
                else:
                    lost.append(tricks - dd_tricks)
                    hit += tricks < c.level + 6
            secs = (time.perf_counter() - t0) / len(boards)
            print(f"{mode:8} {role:9} {statistics.mean(lost):12.3f} {hit / len(boards):8.1%} {secs:8.2f}")


if __name__ == "__main__":
    main()
