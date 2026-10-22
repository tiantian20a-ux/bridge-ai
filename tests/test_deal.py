import random

import pytest

from bridge_ai.cards import Card, Hand
from bridge_ai.deal import Deal, Seat, Vulnerability, dealer_for_board

PBN = "N:AKQ.JT9.876.5432 JT9.876.5432.AKQ 876.5432.AKQ.JT9 5432.AKQ.JT9.876"


def test_random_deal_is_valid_and_reproducible():
    d1 = Deal.random(random.Random(42))
    d2 = Deal.random(random.Random(42))
    assert d1 == d2
    assert sum(len(h) for h in d1.hands) == 52


def test_pbn_roundtrip_and_rotation():
    d = Deal.parse_pbn(PBN)
    assert d.to_pbn() == PBN
    rotated = Deal.parse_pbn("E:" + PBN[2:])
    assert rotated[Seat.EAST] == d[Seat.NORTH]
    assert rotated.to_pbn(Seat.EAST) == "E:" + PBN[2:]
    assert d.owner(Card.parse("SA")) == Seat.NORTH
    assert sum(d[s].hcp() for s in Seat) == 40


def test_invalid_deals():
    h = Hand.parse_pbn("AKQ.JT9.876.5432")
    with pytest.raises(ValueError):
        Deal((h, h, h, h))


def test_seats():
    assert Seat.NORTH.partner == Seat.SOUTH
    assert Seat.WEST.next() == Seat.NORTH
    assert Seat.EAST.same_side(Seat.WEST) and not Seat.EAST.same_side(Seat.SOUTH)


def test_board_vulnerability_and_dealer():
    assert Vulnerability.for_board(1) is Vulnerability.NONE
    assert Vulnerability.for_board(2) is Vulnerability.NS
    assert Vulnerability.for_board(16) is Vulnerability.EW
    assert Vulnerability.for_board(17) is Vulnerability.NONE
    assert dealer_for_board(4) == Seat.WEST
    assert Vulnerability.NS.is_vulnerable(Seat.SOUTH)
    assert not Vulnerability.NS.is_vulnerable(Seat.EAST)
