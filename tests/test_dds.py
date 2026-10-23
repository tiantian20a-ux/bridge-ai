import random

import pytest

from bridge_ai import dds
from bridge_ai.auction import Denomination
from bridge_ai.cards import Card
from bridge_ai.deal import Deal, Seat

pytestmark = pytest.mark.skipif(not dds.available(), reason="DDS library not built (scripts/build_dds.sh)")

# Each hand holds one whole suit: North spades, East hearts, South diamonds, West clubs.
SUITS = Deal.parse_pbn("N:AKQJT98765432... .AKQJT98765432.. ..AKQJT98765432. ...AKQJT98765432")


def test_one_suit_per_hand():
    t = dds.dd_table(SUITS)
    # North declares spades: East leads a heart, North ruffs and draws nothing else to lose.
    assert t[Denomination.SPADES, Seat.NORTH] == 13
    # In notrump the opening leader (declarer's LHO) runs a whole suit.
    assert t[Denomination.NOTRUMP, Seat.NORTH] == 0
    assert t[Denomination.HEARTS, Seat.EAST] == 13
    assert t[Denomination.CLUBS, Seat.NORTH] == 0


def test_table_is_consistent_across_partners_and_sides():
    deal = Deal.random(random.Random(7))
    t = dds.dd_table(deal)
    assert set(t) == {(d, s) for d in Denomination for s in Seat}
    assert all(0 <= v <= 13 for v in t.values())


def test_batch_matches_single():
    rng = random.Random(11)
    deals = [Deal.random(rng) for _ in range(45)]  # spans two CalcAllTables calls
    assert dds.dd_tables(deals) == [dds.dd_table(d) for d in deals]


def test_solve_position_opening_lead_agrees_with_table():
    deal = Deal.random(random.Random(3))
    t = dds.dd_table(deal)
    for denom in Denomination:
        leads = dds.solve_position(deal.hands, denom, Seat.EAST)
        assert set(leads) == set(deal[Seat.EAST])
        # The best lead holds declarer (North) to exactly the table's tricks.
        assert 13 - max(leads.values()) == t[denom, Seat.NORTH]


def test_solve_position_mid_trick():
    remaining = list(SUITS.hands)
    led = Card.parse("HA")
    remaining[Seat.EAST] = remaining[Seat.EAST].remove(led)
    values = dds.solve_position(remaining, Denomination.SPADES, Seat.EAST, [led])
    # South has no hearts and no trumps, so every diamond is equal; NS then lose
    # this trick unless North ruffs, which South cannot affect.
    assert set(values) == set(remaining[Seat.SOUTH])
    assert len(set(values.values())) == 1


def test_solve_positions_matches_single():
    rng = random.Random(5)
    positions = []
    for _ in range(210):  # spans two SolveAllChunksBin calls
        deal = Deal.random(rng)
        positions.append(dds.Position(deal.hands, Denomination(rng.randrange(5)), Seat(rng.randrange(4))))
    batch = dds.solve_positions(positions)
    for pos, got in zip(positions[::20], batch[::20]):
        assert got == dds.solve_position(pos.remaining, pos.trump, pos.leader)
