import json
import os
import pickle
import random
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from bridge_ai import dds
from bridge_ai.auction import Auction, call_from_index, call_index
from bridge_ai.bidding import (AuctionModel, NeuralBidder, PolicyNet, bid_out, encode, legal_mask,
                               load_params)
from bridge_ai.cards import Card, Hand
from bridge_ai.deal import Deal, Seat, Vulnerability
from bridge_ai.play import PIMCPlayer, play_out
from bridge_ai.state import GameState

DATA = Path(__file__).parent / "data"
MODELS = Path(os.environ.get("BRL_MODELS", Path(__file__).parents[1] / "data" / "brl_models"))
VUL = {(False, False): Vulnerability.NONE, (True, False): Vulnerability.NS,
       (False, True): Vulnerability.EW, (True, True): Vulnerability.BOTH}


def random_net(seed=0, scale=0.05):
    r = np.random.default_rng(seed)
    sizes = [(480, 1024), (1024, 1024), (1024, 1024), (1024, 1024), (1024, 38), (1024, 1)]
    return PolicyNet([(r.normal(0, scale, s).astype(np.float32), r.normal(0, scale, s[1]).astype(np.float32))
                      for s in sizes])


def games(name):
    for g in json.load(open(DATA / name)) if name.endswith("obs.json") else [json.load(open(DATA / name))]:
        hands = [Hand.from_cards(Card(c) for c in h) for h in g["hands"]]
        yield g, hands, VUL[g["vul_ns"], g["vul_ew"]]


def test_observation_matches_pgx():
    # Recorded from Pgx 1.4.0 bridge_bidding: indices of the ones in each observation.
    n = 0
    for g, hands, vul in games("pgx_bidding_obs.json"):
        a = Auction(Seat(g["dealer"]))
        for action, obs in zip(g["actions"], g["obs"]):
            assert np.flatnonzero(encode(hands[a.to_act], a, vul)).tolist() == obs
            a.make_call(call_from_index(action))
            n += 1
    assert n > 200


def test_call_probs_is_masked_softmax():
    net = random_net()
    a = Auction.from_calls(Seat.EAST, "1H X")
    hands = [h.mask for h in Deal.random(random.Random(3)).hands]
    p = net.call_probs(a, Vulnerability.EW, hands)
    assert p.shape == (4, 38)
    assert np.allclose(p.sum(axis=1), 1)
    assert (p[:, ~legal_mask(a)] == 0).all()
    for i, m in enumerate(hands):
        z = net.logits(encode(Hand(m), a, Vulnerability.EW)[None])[0][legal_mask(a)]
        q = np.exp(z - z.max())
        assert np.allclose(p[i, legal_mask(a)], q / q.sum(), atol=1e-5)


def test_bidder_makes_legal_calls_to_the_end():
    bidder = NeuralBidder(random_net(1, scale=0.2))
    for seed in range(5):
        deal = Deal.random(random.Random(seed))
        a = bid_out(deal.hands, Seat(seed % 4), Vulnerability.NONE, {s: bidder for s in Seat})
        assert a.is_over


def test_npz_round_trip(tmp_path):
    net = random_net()
    net.save_npz(tmp_path / "m.npz")
    back = PolicyNet.load(tmp_path / "m.npz")
    for (w1, b1), (w2, b2) in zip(net.layers, back.layers):
        assert (w1 == w2).all() and (b1 == b2).all()


def test_model_file_cannot_run_code(tmp_path):
    class Evil:
        def __reduce__(self):
            return (os.system, ("true",))

    path = tmp_path / "evil.pkl"
    path.write_bytes(pickle.dumps({"actor_critic/linear": {"w": Evil()}}))
    with pytest.raises(pickle.UnpicklingError):
        load_params(path)


def test_auction_model_weights():
    net = random_net()
    model = AuctionModel(net, Vulnerability.NONE)
    a = Auction.from_calls(Seat.NORTH, "1S P 2S P P P")
    hands = {Seat.NORTH: [h.mask for h in Deal.random(random.Random(5)).hands]}
    w = model.weights(a, hands)
    assert w.shape == (4,) and w.max() == 1 and (w > 0).all()
    # Only North's own calls count: 1S and the final pass.
    expect = np.ones(4)
    prefix = Auction(Seat.NORTH)
    for call in a.calls:
        if prefix.to_act is Seat.NORTH:
            expect *= np.maximum(net.call_probs(prefix, Vulnerability.NONE, hands[Seat.NORTH])[:, call_index(call)], 1e-3)
        prefix.make_call(call)
    assert np.allclose(w, expect / expect.max(), rtol=1e-4)


@pytest.mark.skipif(not dds.available(), reason="DDS library not built (scripts/build_dds.sh)")
def test_pimc_with_auction_model_plays_out():
    deal = Deal.random(random.Random(7))
    g = GameState(deal)
    for call in Auction.from_calls(Seat.NORTH, "1NT P 3NT P P P").calls:
        g.make_call(call)
    agent = PIMCPlayer(8, rng=random.Random(0), auction_model=AuctionModel(random_net(), Vulnerability.NONE))
    tricks = play_out(g, {s: agent for s in Seat})
    assert 0 <= tricks <= 13


sl_model = pytest.mark.skipif(not (MODELS / "model-sl.pkl").exists(),
                              reason="brl models not downloaded (scripts/fetch_brl_models.sh)")


@sl_model
def test_released_model_logits_match_brl():
    net = PolicyNet.load(MODELS / "model-sl.pkl")
    ((g, hands, vul),) = games("brl_sl_logits.json")
    a = Auction(Seat(g["dealer"]))
    for action, logits in zip(g["actions"], g["logits"]):
        assert np.allclose(net.logits(encode(hands[a.to_act], a, vul)[None])[0], logits, atol=1e-3)
        a.make_call(call_from_index(action))


@sl_model
def test_released_model_reads_a_spade_opening():
    net = PolicyNet.load(MODELS / "model-sl.pkl")
    model = AuctionModel(net, Vulnerability.NONE)
    long_spades = Hand.parse_pbn("AKJ76.K32.Q4.J82").mask
    short_spades = Hand.parse_pbn("A.KJ32.Q4.KJ8765").mask
    w = model.weights(Auction.from_calls(Seat.NORTH, "1S"), {Seat.NORTH: [long_spades, short_spades]})
    assert w[0] > 20 * w[1]
