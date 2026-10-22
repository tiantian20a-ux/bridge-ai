"""Check bridge_ai.dds against endplay's independent DDS build on random deals.

For each deal it compares the full double-dummy table (5 denominations x 4
declarers), then plays a random number of random legal cards and compares the
per-card values from solve_position at that point.

Usage: python scripts/verify_dds.py [--deals 1000] [--seed 20261022]
Needs: pip install endplay; bash scripts/build_dds.sh
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import endplay.dds as edds
import endplay.types as et

from bridge_ai import dds
from bridge_ai.auction import Denomination
from bridge_ai.cards import Card, Rank, Suit
from bridge_ai.deal import Deal, Seat
from bridge_ai.state import trump_suit

_E_DENOM = {Denomination.CLUBS: et.Denom.clubs, Denomination.DIAMONDS: et.Denom.diamonds,
            Denomination.HEARTS: et.Denom.hearts, Denomination.SPADES: et.Denom.spades,
            Denomination.NOTRUMP: et.Denom.nt}
_E_PLAYER = {Seat.NORTH: et.Player.north, Seat.EAST: et.Player.east,
             Seat.SOUTH: et.Player.south, Seat.WEST: et.Player.west}


def reference_table(deal: Deal) -> dict:
    t = edds.calc_dd_table(et.Deal(deal.to_pbn()))
    return {(d, s): t[_E_DENOM[d], _E_PLAYER[s]] for d in Denomination for s in Seat}


def reference_position(remaining, denom, leader, trick) -> dict[Card, int]:
    hands = list(remaining)
    for k, c in enumerate(trick):  # endplay plays the trick's cards from the hands itself
        hands[leader.next(k)] = hands[leader.next(k)].add(c)
    pbn = "N:" + " ".join(h.to_pbn() for h in hands)
    d = et.Deal(pbn, first=_E_PLAYER[leader], trump=_E_DENOM[denom])
    for c in trick:
        d.play(str(c))
    # endplay suits are S, H, D, C = 0..3 and ranks are bit values (deuce = 1 << 2)
    return {Card.of(Suit(3 - int(c.suit)), Rank(int(c.rank).bit_length() - 3)): v
            for c, v in edds.solve_board(d)}


def random_position(rng: random.Random, deal: Deal):
    """Play 0..47 random legal cards; return (remaining hands, denomination, trick leader, trick)."""
    denom = rng.choice(list(Denomination))
    trump = trump_suit(denom)
    remaining = list(deal.hands)
    leader = Seat(rng.randrange(4))
    trick: list[Card] = []
    for _ in range(rng.randrange(48)):
        seat = leader.next(len(trick))
        hand = remaining[seat]
        follow = hand.cards_in(trick[0].suit) if trick else []
        card = rng.choice(follow or list(hand))
        remaining[seat] = hand.remove(card)
        trick.append(card)
        if len(trick) == 4:
            def strength(k: int) -> int:
                c = trick[k]
                if trump is not None and c.suit == trump:
                    return 100 + c.rank
                return c.rank if c.suit == trick[0].suit else -1
            leader = leader.next(max(range(4), key=strength))
            trick = []
    return remaining, denom, leader, trick


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    deals = [Deal.random(rng) for _ in range(args.deals)]

    t0 = time.perf_counter()
    ours = dds.dd_tables(deals)
    t_ours = time.perf_counter() - t0
    t0 = time.perf_counter()
    refs = [reference_table(d) for d in deals]
    t_ref = time.perf_counter() - t0
    table_bad = [i for i, (a, b) in enumerate(zip(ours, refs)) if a != b]

    pos_bad = []
    for i, deal in enumerate(deals):
        remaining, denom, leader, trick = random_position(rng, deal)
        a = dds.solve_position(remaining, denom, leader, trick)
        b = reference_position(remaining, denom, leader, trick)
        if a != b:
            pos_bad.append(i)

    n = len(deals)
    print(f"DD tables: {n - len(table_bad)}/{n} deals identical "
          f"(ours {t_ours:.1f}s batched, endplay {t_ref:.1f}s one by one)")
    print(f"Positions: {n - len(pos_bad)}/{n} identical per-card values")
    for i in (table_bad + pos_bad)[:5]:
        print("  mismatch on deal", i, deals[i].to_pbn())
    return 0 if not table_bad and not pos_bad else 1


if __name__ == "__main__":
    sys.exit(main())
