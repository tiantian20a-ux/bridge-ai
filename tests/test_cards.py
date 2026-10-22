import pytest

from bridge_ai.cards import Card, Hand, Rank, Suit


def test_card_roundtrip_all_52():
    assert {str(Card(i)) for i in range(52)} == {s + r for s in "CDHS" for r in "23456789TJQKA"}
    for i in range(52):
        assert Card.parse(str(Card(i))) == Card(i)


def test_card_parse_ten_and_order():
    assert Card.parse("D10") == Card.of(Suit.DIAMONDS, Rank.TEN)
    assert Card.parse("SA") > Card.parse("SK") > Card.parse("HA")


def test_card_out_of_range():
    with pytest.raises(ValueError):
        Card(52)


def test_hand_pbn_roundtrip_and_stats():
    h = Hand.parse_pbn("AKQ.JT9.876.5432")
    assert len(h) == 13
    assert h.to_pbn() == "AKQ.JT9.876.5432"
    assert h.shape() == (3, 3, 3, 4)
    assert h.hcp() == 10
    assert Card.parse("SA") in h and Card.parse("CA") not in h


def test_hand_void_suit():
    h = Hand.parse_pbn("AKQJT98765432...")
    assert h.shape() == (13, 0, 0, 0)
    assert h.to_pbn() == "AKQJT98765432..."


def test_hand_add_remove_duplicates():
    h = Hand.from_cards([Card.parse("H2")])
    with pytest.raises(ValueError):
        h.add(Card.parse("H2"))
    with pytest.raises(ValueError):
        h.remove(Card.parse("H3"))
    assert len(h.remove(Card.parse("H2"))) == 0
    with pytest.raises(ValueError):
        Hand.from_cards([Card(0), Card(0)])
