import random

import pytest

from bridge_ai.auction import Auction, parse_call
from bridge_ai.cards import Card, Hand, Suit
from bridge_ai.deal import Deal, Seat
from bridge_ai.inference import constraints_from_auction
from bridge_ai.sampling import HandConstraint, Sampler, View
from bridge_ai.state import GameState

PBN = "N:AKQ.JT9.876.5432 JT9.876.5432.AKQ 876.5432.AKQ.JT9 5432.AKQ.JT9.876"


def game(calls="1NT P P P", pbn=PBN):
    g = GameState(Deal.parse_pbn(pbn))
    for c in calls.split():
        g.make_call(parse_call(c))
    return g


def test_view_before_and_after_opening_lead():
    g = game()  # North declares, East leads, South is dummy
    v = View.from_state(g, Seat.EAST)
    assert v.hidden == [Seat.NORTH, Seat.SOUTH, Seat.WEST]
    assert len(v.unseen) == 39
    g.play(Card.parse("CA"))
    v = View.from_state(g, Seat.WEST)
    assert v.hidden == [Seat.NORTH, Seat.EAST]
    assert len(v.unseen) == 25  # 52 - West 13 - dummy 13 - the lead
    assert v.counts[Seat.EAST] == 12


def test_samples_are_consistent_with_the_play():
    rng = random.Random(2)
    g = game()
    for card in ["CA", "C9", "C6", "C2", "CK"]:  # East wins the first trick and leads again
        g.play(Card.parse(card))
    v = View.from_state(g, Seat.EAST)
    for layout in Sampler(v, rng=rng).sample(50):
        union = 0
        for s in Seat:
            assert len(layout[s]) == len(g.remaining[s])
            assert not union & layout[s].mask
            union |= layout[s].mask
        assert union == sum(h.mask for h in g.remaining)
        assert layout[Seat.EAST] == g.remaining[Seat.EAST]


def test_voids_are_respected():
    rng = random.Random(3)
    deal = Deal.random(rng)
    g = GameState(deal)
    for c in "1NT P P P".split():
        g.make_call(parse_call(c))
    while len(g.tricks) < 6:
        g.play(rng.choice(g.legal_plays()))
    v = View.from_state(g, Seat.EAST)
    for layout in Sampler(v, rng=rng).sample(100):
        for s in Seat:
            for suit in v.voids[s]:
                assert layout[s].suit_length(suit) == 0


def test_sampling_is_uniform():
    # After the opening lead, West sees 25 unseen cards split 13 (North) / 12 (East):
    # any one card is with North 13/25 of the time.
    rng = random.Random(4)
    g = game()
    g.play(Card.parse("CA"))
    v = View.from_state(g, Seat.WEST)
    n = 4000
    card = next(iter(v.unseen))
    hits = sum(card in layout[Seat.NORTH] for layout in Sampler(v, rng=rng).sample(n))
    assert abs(hits / n - 13 / 25) < 0.03


def test_split_frequencies_match_combinatorics():
    # 3 hidden seats before the lead; East's view: hearts held by the others are 13 - 3 = 10.
    # The chance North holds all 3 of some specific cards is C(36,10)/C(39,13).
    rng = random.Random(5)
    g = game()
    v = View.from_state(g, Seat.EAST)
    cards = list(v.unseen)[:2]
    n = 6000
    both = sum(all(c in l[Seat.NORTH] for c in cards) for l in Sampler(v, rng=rng).sample(n))
    assert abs(both / n - (13 * 12) / (39 * 38)) < 0.02


def test_hcp_and_length_constraints():
    rng = random.Random(6)
    g = game("P 1NT P P P", pbn="N:AKQ.JT9.876.5432 JT9.876.5432.AKQ 876.5432.AKQ.JT9 5432.AKQ.JT9.876")
    # East opened 1NT and declares; South leads, West is dummy.
    cons = constraints_from_auction(g.auction)
    assert cons[Seat.EAST].hcp == (15, 17)
    v = View.from_state(g, Seat.SOUTH)
    s = Sampler(v, cons, rng)
    for layout in s.sample(200):
        assert 15 <= layout[Seat.EAST].hcp() <= 17
        assert all(2 <= n <= 5 for n in layout[Seat.EAST].shape())
        assert layout[Seat.NORTH].hcp() <= 11
    assert not s.relaxed


def test_contradictory_constraints_are_dropped():
    g = game()
    g.play(Card.parse("CA"))
    v = View.from_state(g, Seat.WEST)
    impossible = HandConstraint(lengths=((13, 13),) * 4)
    s = Sampler(v, {Seat.EAST: impossible}, random.Random(7))
    assert s.relaxed
    assert len(s.sample(5)) == 5


def test_unreachable_points_fall_back():
    g = game()
    g.play(Card.parse("CA"))
    v = View.from_state(g, Seat.WEST)
    s = Sampler(v, {Seat.EAST: HandConstraint(hcp=(37, 37))}, random.Random(8))
    assert len(s.sample(3, max_tries=10)) == 3
    assert s.relaxed


def test_auction_rules():
    c = constraints_from_auction(Auction.from_calls(Seat.NORTH, "P 1H 2C P"))
    assert c[Seat.NORTH].hcp == (0, 11)
    assert c[Seat.EAST].lengths[Suit.HEARTS] == (5, 13)
    assert c[Seat.SOUTH].lengths[Suit.CLUBS] == (5, 13) and c[Seat.SOUTH].hcp == (8, 17)
    assert Seat.WEST not in c
    c = constraints_from_auction(Auction.from_calls(Seat.NORTH, "2S 3H"))
    assert c[Seat.NORTH].lengths[Suit.SPADES] == (6, 6)
    assert c[Seat.EAST].lengths[Suit.HEARTS] == (5, 13)  # 3H is the cheapest heart bid over 2S
    c = constraints_from_auction(Auction.from_calls(Seat.NORTH, "1D 2S"))
    assert c[Seat.EAST].lengths[Suit.SPADES] == (6, 13)  # weak jump overcall
