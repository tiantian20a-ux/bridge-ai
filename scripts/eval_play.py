"""Measure card-play agents against double-dummy play.

On random deals, the side with more high-card points declares the contract
that double dummy says it makes exactly: its best strain, at the level of
its double-dummy tricks. Each agent then plays one side while a
double-dummy player (who sees all hands) plays the other:

- as declarer (dummy included), against double-dummy defenders;
- as both defenders, against a double-dummy declarer.

Perfect opponents hold the agent to at most the double-dummy result, so
"tricks lost" (0 is perfect) and the share of contracts made or beaten
measure how close it gets to perfect play. (Double-dummy declarers always
make, so the defence is measured by overtricks given away.)

    python scripts/eval_play.py --deals 200 --agents pimc simple random
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from bridge_ai import dds
from bridge_ai.auction import Bid, Denomination, Pass
from bridge_ai.deal import Deal, Seat
from bridge_ai.play import DoubleDummyPlayer, PIMCPlayer, RandomPlayer, SimplePlayer, play_out
from bridge_ai.state import GameState


def contract_for(deal: Deal, table: dds.DDTable) -> tuple[Seat, Bid, int] | None:
    """Declarer, contract and double-dummy tricks, or None if the side makes nothing."""
    ns = deal[Seat.NORTH].hcp() + deal[Seat.SOUTH].hcp()
    side = (Seat.NORTH, Seat.SOUTH) if ns >= 20 else (Seat.EAST, Seat.WEST)
    # Prefer more tricks, then notrump and majors (the strains that score better).
    order = [Denomination.CLUBS, Denomination.DIAMONDS, Denomination.HEARTS, Denomination.SPADES, Denomination.NOTRUMP]
    tricks, _, denom, declarer = max((table[d, s], order.index(d), d, s) for d in Denomination for s in side)
    if tricks < 7:
        return None
    return declarer, Bid(tricks - 6, denom), tricks


def new_game(deal: Deal, declarer: Seat, bid: Bid) -> GameState:
    g = GameState(Deal(deal.hands, dealer=declarer))
    for call in (bid, Pass, Pass, Pass):
        g.make_call(call)
    return g


def make_agent(name: str, samples: int, rng: random.Random):
    if name == "pimc":
        return PIMCPlayer(samples, use_auction=False, rng=rng)
    if name == "simple":
        return SimplePlayer()
    if name == "random":
        return RandomPlayer(rng)
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--samples", type=int, default=64, help="PIMC layouts per decision")
    ap.add_argument("--agents", nargs="+", default=["pimc", "simple", "random"])
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    boards = []
    while len(boards) < args.deals:
        deal = Deal.random(rng)
        c = contract_for(deal, dds.dd_table(deal))
        if c:
            boards.append((deal, *c))

    dd = DoubleDummyPlayer()
    print(f"{args.deals} deals, PIMC {args.samples} samples per decision")
    print(f"{'agent':8} {'role':9} {'tricks lost':>12} {'made':>11} {'s/board':>8}")
    for name in args.agents:
        for role in ("declarer", "defence"):
            agent = make_agent(name, args.samples, random.Random(args.seed))
            lost, hit, t0 = [], 0, time.perf_counter()
            for deal, declarer, bid, dd_tricks in boards:
                g = new_game(deal, declarer, bid)
                ours = {declarer, declarer.partner} if role == "declarer" else {declarer.next(), declarer.next(3)}
                players = {s: agent if s in ours else dd for s in Seat}
                tricks = play_out(g, players)
                if role == "declarer":
                    lost.append(dd_tricks - tricks)
                    hit += tricks >= bid.level + 6
                else:
                    lost.append(tricks - dd_tricks)
            secs = (time.perf_counter() - t0) / len(boards)
            made = f"{hit / len(boards):.1%}" if role == "declarer" else "-"
            print(f"{name:8} {role:9} {statistics.mean(lost):12.3f} {made:>11} {secs:8.2f}")


if __name__ == "__main__":
    main()
