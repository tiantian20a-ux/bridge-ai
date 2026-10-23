import random

import pytest

from bridge_ai import dds
from bridge_ai.auction import parse_call
from bridge_ai.cards import Card
from bridge_ai.deal import Deal, Seat
from bridge_ai.play import DoubleDummyPlayer, PIMCPlayer, RandomPlayer, SimplePlayer, play_out
from bridge_ai.state import GameState, Phase

pytestmark = pytest.mark.skipif(not dds.available(), reason="DDS library not built (scripts/build_dds.sh)")


def game(deal, calls):
    g = GameState(deal)
    for c in calls.split():
        g.make_call(parse_call(c))
    return g


def test_agents_play_legal_full_boards():
    rng = random.Random(1)
    agents = [PIMCPlayer(8, rng=rng), SimplePlayer(), RandomPlayer(rng), DoubleDummyPlayer()]
    for i in range(4):
        g = game(Deal.random(rng), "1NT P P P")
        players = {s: agents[(i + s) % 4] for s in Seat}
        tricks = play_out(g, players)
        assert g.phase is Phase.DONE and 0 <= tricks <= 13


def test_double_dummy_play_reaches_the_table_value():
    rng = random.Random(2)
    dd = DoubleDummyPlayer()
    for _ in range(3):
        deal = Deal.random(rng)
        g = game(deal, "1S P P P")
        assert play_out(g, {s: dd for s in Seat}) == dds.dd_table(deal)[3, Seat.NORTH]


def test_pimc_against_perfect_defence_never_beats_double_dummy():
    rng = random.Random(3)
    dd = DoubleDummyPlayer()
    pimc = PIMCPlayer(16, use_auction=False, rng=rng)
    for _ in range(3):
        deal = Deal.random(rng)
        g = game(deal, "1NT P P P")
        tricks = play_out(g, {Seat.NORTH: pimc, Seat.EAST: dd, Seat.WEST: dd})
        assert tricks <= dds.dd_table(deal)[4, Seat.NORTH]


def test_pimc_cashes_a_sure_winner():
    # Spades are trumps. North has the last trump and three diamonds no one can beat
    # (East and West hold no diamonds), so every sample agrees.
    deal = Deal.parse_pbn("N:AKQJT98765432... .AKQJT98765432.. ..AKQJT98765432. ...AKQJT98765432")
    g = game(deal, "7S P P P")  # North declares, East leads
    g.play(Card.parse("HA"))
    pimc = PIMCPlayer(8, rng=random.Random(4))
    card = pimc.choose(g)  # South (dummy) has only diamonds: any one is equal
    assert card.suit == Card.parse("D2").suit
    g.play(card)
    g.play(Card.parse("C2"))
    assert pimc.choose(g).suit == Card.parse("S2").suit  # North ruffs
    assert pimc.last.make == 1.0


def test_simple_player_second_hand_low_third_hand_high():
    deal = Deal.parse_pbn("N:AKQ.JT9.876.5432 JT9.876.5432.AKQ 876.5432.AKQ.JT9 5432.AKQ.JT9.876")
    g = game(deal, "1NT P P P")  # East leads
    g.play(Card.parse("CA"))
    assert str(SimplePlayer().choose(g)) == "C9"  # second hand low
    g.play(Card.parse("C9"))
    assert str(SimplePlayer().choose(g)) == "C6"  # partner of leader: East is winning, play low
