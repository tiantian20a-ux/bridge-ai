import pytest

from bridge_ai.auction import Contract, Denomination
from bridge_ai.deal import Seat, Vulnerability
from bridge_ai.scoring import contract_score, imps, score_ns

C, D, H, S, NT = (Denomination.CLUBS, Denomination.DIAMONDS, Denomination.HEARTS,
                  Denomination.SPADES, Denomination.NOTRUMP)


@pytest.mark.parametrize("level, denom, doubled, tricks, vul, expected", [
    # partscores and games
    (1, C, 0, 7, False, 70),
    (2, H, 0, 8, False, 110),
    (1, NT, 0, 7, False, 90),
    (1, NT, 0, 9, False, 150),
    (3, NT, 0, 9, False, 400),
    (3, NT, 0, 9, True, 600),
    (4, S, 0, 10, False, 420),
    (4, H, 0, 11, True, 650),
    (5, D, 0, 11, False, 400),
    (5, C, 0, 11, True, 600),
    # slams
    (6, S, 0, 12, False, 980),
    (6, NT, 0, 12, True, 1440),
    (7, NT, 0, 13, False, 1520),
    (7, C, 0, 13, True, 2140),
    # doubled and redoubled, made
    (1, C, 1, 7, False, 140),
    (2, S, 1, 8, False, 470),
    (2, S, 1, 9, True, 870),
    (1, NT, 2, 7, False, 560),
    (1, NT, 2, 8, True, 1160),
    (4, S, 1, 10, True, 790),
    (7, NT, 2, 13, True, 2980),
    # down undoubled
    (4, S, 0, 9, False, -50),
    (4, S, 0, 7, True, -300),
    # down doubled, not vulnerable: 100, 300, 500, then +300
    (4, S, 1, 9, False, -100),
    (4, S, 1, 8, False, -300),
    (4, S, 1, 7, False, -500),
    (4, S, 1, 6, False, -800),
    (7, NT, 1, 0, False, -3500),
    # down doubled, vulnerable: 200, 500, 800, then +300
    (4, S, 1, 9, True, -200),
    (4, S, 1, 8, True, -500),
    (4, S, 1, 7, True, -800),
    (7, NT, 1, 0, True, -3800),
    # down redoubled
    (4, S, 2, 7, False, -1000),
    (4, S, 2, 8, True, -1000),
])
def test_contract_score(level, denom, doubled, tricks, vul, expected):
    assert contract_score(level, denom, doubled, tricks, vul) == expected


def test_score_ns_sign_and_vulnerability():
    vul = Vulnerability.NS
    assert score_ns(Contract(4, S, Seat.SOUTH), 10, vul) == 620
    assert score_ns(Contract(4, S, Seat.EAST), 10, vul) == -420
    assert score_ns(Contract(4, S, Seat.WEST, doubled=1), 9, vul) == 100
    assert score_ns(None, 0, vul) == 0


def test_bad_arguments():
    with pytest.raises(ValueError):
        contract_score(0, C, 0, 7, False)
    with pytest.raises(ValueError):
        contract_score(1, C, 3, 7, False)
    with pytest.raises(ValueError):
        contract_score(1, C, 0, 14, False)


@pytest.mark.parametrize("diff, expected", [
    (0, 0), (10, 0), (20, 1), (40, 1), (50, 2), (80, 2), (90, 3), (120, 3),
    (130, 4), (170, 5), (210, 5), (220, 6), (270, 7), (320, 8), (370, 9),
    (420, 9), (430, 10), (490, 10), (500, 11), (590, 11), (600, 12), (740, 12),
    (750, 13), (900, 14), (1100, 15), (1300, 16), (1500, 17), (1750, 18),
    (2000, 19), (2250, 20), (2500, 21), (3000, 22), (3500, 23), (3990, 23),
    (4000, 24), (7600, 24),
])
def test_imps(diff, expected):
    assert imps(diff) == expected
    assert imps(-diff) == -expected
