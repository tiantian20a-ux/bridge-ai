import pytest

from bridge_ai.auction import (
    NUM_CALLS, Auction, Bid, Denomination, Double, IllegalCall, Pass, Redouble,
    call_from_index, call_index, parse_call,
)
from bridge_ai.deal import Seat


def test_call_index_roundtrip():
    for i in range(NUM_CALLS):
        assert call_index(call_from_index(i)) == i
    assert str(call_from_index(37)) == "7NT"


def test_passed_out():
    a = Auction.from_calls(Seat.NORTH, "P P P P")
    assert a.is_over and a.is_passed_out and a.contract() is None


def test_simple_contract_and_declarer():
    # N opens 1H, S raises; N was first to bid hearts, so N declares.
    a = Auction.from_calls(Seat.NORTH, "1H P 2H P 4H P P P")
    c = a.contract()
    assert c.level == 4 and c.denomination is Denomination.HEARTS
    assert c.declarer == Seat.NORTH and c.doubled == 0


def test_declarer_is_first_of_side_to_name_strain():
    a = Auction.from_calls(Seat.NORTH, "1C P 1S P 2NT P 3NT P P P")
    assert a.contract().declarer == Seat.NORTH
    a = Auction.from_calls(Seat.NORTH, "1C P 1NT P 3NT P P P")
    assert a.contract().declarer == Seat.SOUTH


def test_doubled_and_redoubled():
    a = Auction.from_calls(Seat.EAST, "1S X P P P")
    assert str(a.contract()) == "1SX by E"
    a = Auction.from_calls(Seat.EAST, "1S X XX P P P")
    assert a.contract().doubled == 2


def test_double_cancelled_by_new_bid():
    a = Auction.from_calls(Seat.NORTH, "1H X 2H P P P")
    assert a.contract().doubled == 0


def test_illegal_calls():
    a = Auction.from_calls(Seat.NORTH, "1H")
    with pytest.raises(IllegalCall):
        a.make_call(Bid(1, Denomination.DIAMONDS))  # insufficient
    with pytest.raises(IllegalCall):
        a.make_call(Redouble)  # nothing to redouble
    a.make_call(Pass)
    with pytest.raises(IllegalCall):
        a.make_call(Double)  # cannot double partner
    assert not Auction(Seat.NORTH).is_legal(Double)


def test_legal_calls_and_not_over_early():
    a = Auction(Seat.NORTH)
    assert len(a.legal_calls()) == 36  # pass + 35 bids
    a = Auction.from_calls(Seat.NORTH, "P P P")
    assert not a.is_over
    a = Auction.from_calls(Seat.NORTH, "1C P P")
    assert not a.is_over and a.to_act == Seat.WEST
    a.make_call(Pass)
    assert a.is_over and not a.is_legal(Pass)


def test_parse_call_aliases():
    assert parse_call("pass") is Pass
    assert parse_call("dbl") is Double
    assert parse_call("1N") == Bid(1, Denomination.NOTRUMP)
