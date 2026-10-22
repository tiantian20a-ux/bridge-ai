import random

import pytest

from bridge_ai.auction import parse_call
from bridge_ai.cards import Card
from bridge_ai.deal import Deal, Seat
from bridge_ai.state import GameState, IllegalPlay, Phase, Trick

PBN = "N:AKQ.JT9.876.5432 JT9.876.5432.AKQ 876.5432.AKQ.JT9 5432.AKQ.JT9.876"


def bid(state, calls):
    for c in calls.split():
        state.make_call(parse_call(c))


def test_auction_to_play_transition():
    g = GameState(Deal.parse_pbn(PBN))
    assert g.phase is Phase.AUCTION and g.to_act == Seat.NORTH
    bid(g, "1S P P P")
    assert g.phase is Phase.PLAY
    assert g.declarer == Seat.NORTH and g.dummy == Seat.SOUTH
    assert g.to_act == Seat.EAST  # opening leader is declarer's LHO


def test_passed_out_game_is_done():
    g = GameState(Deal.parse_pbn(PBN))
    bid(g, "P P P P")
    assert g.phase is Phase.DONE


def test_must_follow_suit():
    g = GameState(Deal.parse_pbn(PBN))
    bid(g, "1NT P P P")
    g.play(Card.parse("CA"))  # East leads a club
    legal = g.legal_plays()  # South has JT9 of clubs
    assert set(map(str, legal)) == {"CJ", "CT", "C9"}
    with pytest.raises(IllegalPlay):
        g.play(Card.parse("DA"))


def test_trick_winner_with_trump():
    t = Trick(Seat.EAST, [Card.parse(c) for c in ["CA", "CJ", "C8", "S2"]])
    assert t.winner(trump=None) == Seat.EAST
    from bridge_ai.cards import Suit
    assert t.winner(trump=Suit.SPADES) == Seat.NORTH


def test_random_full_game_plays_out():
    rng = random.Random(7)
    for _ in range(20):
        g = GameState(Deal.random(rng))
        bid(g, "1NT P P P")
        while g.phase is Phase.PLAY:
            g.play(rng.choice(g.legal_plays()))
        assert len(g.tricks) == 13
        assert g.tricks_won(Seat.NORTH) + g.tricks_won(Seat.EAST) == 13
        assert all(len(h) == 0 for h in g.remaining)
