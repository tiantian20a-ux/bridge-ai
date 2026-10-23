"""Neural bidding: the brl policy network run with numpy.

brl (github.com/harukaki/brl, Apache-2.0) trains a 4x1024 MLP on the Pgx
``bridge_bidding`` environment. This module reads its released parameter
files without JAX, builds the same 480-number observation from our
``Auction`` and ``Hand``, and turns the network into

- ``NeuralBidder``, which makes calls (the most likely legal call, as brl
  does when it evaluates), and
- ``PolicyNet.call_probs``, the chance the network gives each call, which
  ``AuctionModel`` uses to weigh card-play samples by how well each layout
  explains the auction.

Observation layout (Pgx 1.4.0, the OpenSpiel encoding), from the view of
the seat to act:

- 4: [we not vulnerable, we vulnerable, they not vulnerable, they vulnerable]
- 4: passes before the opening bid, by relative seat (0 = self, 1 = LHO, ...)
- 420: for each of the 35 bids, who made it, who doubled it and who
  redoubled it (3 x 4 relative seats)
- 52: own hand, card index suit + 4 * rank (clubs = 0, two = 0)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .auction import NUM_CALLS, Auction, Bid, Call, Double, Pass, call_from_index, call_index
from .cards import Hand
from .deal import Seat, Vulnerability

OBS_SIZE = 480
HAND_OFFSET = 428

# Our card index (suit * 13 + rank) -> position of that card in the observation's hand block.
_CARD_TO_OBS = np.array([HAND_OFFSET + (i // 13) + 4 * (i % 13) for i in range(52)])
_BITS = np.arange(52, dtype=np.uint64)


def hand_bits(masks: Sequence[int]) -> np.ndarray:
    """(n, 52) 0/1 array of the cards in each 52-bit hand mask, by our card index."""
    m = np.array(masks, dtype=np.uint64)[:, None]
    return ((m >> _BITS) & np.uint64(1)).astype(np.float32)


def encode_history(auction: Auction, vulnerability: Vulnerability) -> np.ndarray:
    """The observation without the hand, for the seat to act after ``auction``."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    me = auction.to_act
    us, them = vulnerability.is_vulnerable(me), vulnerability.is_vulnerable(me.next())
    obs[0 + us] = 1
    obs[2 + them] = 1
    last_bid: Bid | None = None
    for i, call in enumerate(auction.calls):
        rel = (auction.dealer.next(i) - me) % 4
        if isinstance(call, Bid):
            last_bid = call
            obs[8 + 12 * call.index + rel] = 1
        elif call is Pass:
            if last_bid is None:
                obs[4 + rel] = 1
        else:
            assert last_bid is not None
            obs[8 + 12 * last_bid.index + (4 if call is Double else 8) + rel] = 1
    return obs


def encode(hand: Hand, auction: Auction, vulnerability: Vulnerability) -> np.ndarray:
    obs = encode_history(auction, vulnerability)
    obs[_CARD_TO_OBS[[c.index for c in hand]]] = 1
    return obs


def legal_mask(auction: Auction) -> np.ndarray:
    return np.array([auction.is_legal(call_from_index(i)) for i in range(NUM_CALLS)])


class _ParamUnpickler(pickle.Unpickler):
    """Reads brl's pickled JAX arrays as numpy arrays, and allows nothing else."""

    def find_class(self, module: str, name: str):
        if module == "jax._src.array" and name == "_reconstruct_array":
            return _reconstruct_array
        if (module, name) in {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct")}:
            return np.ndarray.__reduce__(np.zeros(0))[0]
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        raise pickle.UnpicklingError(f"unexpected object in model file: {module}.{name}")


def _reconstruct_array(fun, args, arr_state, aval_state):
    value = fun(*args)
    value.__setstate__(arr_state)
    return value


def load_params(path: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """(weight, bias) per layer from a brl ``.pkl`` or an ``.npz`` with w0, b0, w1, ..."""
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as z:
            n = len(z.files) // 2
            return [(z[f"w{i}"], z[f"b{i}"]) for i in range(n)]
    with path.open("rb") as f:
        tree = _ParamUnpickler(f).load()
    names = sorted(tree, key=lambda k: int(k.rsplit("_", 1)[1]) if k[-1].isdigit() else 0)
    return [(np.asarray(tree[k]["w"], np.float32), np.asarray(tree[k]["b"], np.float32)) for k in names]


@dataclass
class PolicyNet:
    """brl's "DeepMind" network: four 1024-unit ReLU layers, then policy and value heads."""

    layers: list[tuple[np.ndarray, np.ndarray]]

    @classmethod
    def load(cls, path: str | Path) -> PolicyNet:
        layers = load_params(path)
        if len(layers) != 6 or layers[0][0].shape != (OBS_SIZE, 1024):
            raise ValueError(f"{path}: not a brl DeepMind-type model")
        return cls(layers)

    def save_npz(self, path: str | Path) -> None:
        np.savez(path, **{f"{k}{i}": a for i, layer in enumerate(self.layers) for k, a in zip("wb", layer)})

    def _hidden(self, obs: np.ndarray) -> np.ndarray:
        x = obs
        for w, b in self.layers[:4]:
            x = np.maximum(x @ w + b, 0)
        return x

    def logits(self, obs: np.ndarray) -> np.ndarray:
        """(n, 38) call logits for (n, 480) observations."""
        w, b = self.layers[4]
        return self._hidden(obs) @ w + b

    def value(self, obs: np.ndarray) -> np.ndarray:
        w, b = self.layers[5]
        return (self._hidden(obs) @ w + b)[:, 0]

    def call_probs(self, auction: Auction, vulnerability: Vulnerability, hands: Sequence[int]) -> np.ndarray:
        """(n, 38) chance of each legal call by the seat to act, for each of its possible hands (masks)."""
        obs = np.repeat(encode_history(auction, vulnerability)[None], len(hands), axis=0)
        obs[:, HAND_OFFSET:] = hand_bits(hands)[:, np.argsort(_CARD_TO_OBS)]
        z = self.logits(obs)
        z[:, ~legal_mask(auction)] = -np.inf
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)


class NeuralBidder:
    """Makes the call the network rates highest among the legal ones."""

    def __init__(self, net: PolicyNet):
        self.net = net

    def choose(self, hand: Hand, auction: Auction, vulnerability: Vulnerability) -> Call:
        z = self.net.logits(encode(hand, auction, vulnerability)[None])[0]
        z[~legal_mask(auction)] = -np.inf
        return call_from_index(int(np.argmax(z)))


def bid_out(hands: Sequence[Hand], dealer: Seat, vulnerability: Vulnerability,
            bidders: dict[Seat, NeuralBidder]) -> Auction:
    """Run a whole auction, each seat calling with its own bidder."""
    auction = Auction(dealer)
    while not auction.is_over:
        seat = auction.to_act
        auction.make_call(bidders[seat].choose(hands[seat], auction, vulnerability))
    return auction


@dataclass
class AuctionModel:
    """How well a layout of hidden hands explains the auction, under a bidding network.

    The weight of a layout is the product, over every call a hidden seat
    made, of the chance the network gives that call with that seat's
    hand. It assumes the other players bid like the network.
    """

    net: PolicyNet
    vulnerability: Vulnerability
    floor: float = 1e-3  # lowest chance counted for one call, so one odd call cannot rule a layout out

    def weights(self, auction: Auction, hands: dict[Seat, Sequence[int]]) -> np.ndarray:
        """Weight of each layout; ``hands[seat][k]`` is the original hand (mask) of ``seat`` in layout k."""
        n = len(next(iter(hands.values())))
        logw = np.zeros(n)
        prefix = Auction(auction.dealer)
        for call in auction.calls:
            seat = prefix.to_act
            if seat in hands:
                p = self.net.call_probs(prefix, self.vulnerability, hands[seat])[:, call_index(call)]
                logw += np.log(np.maximum(p, self.floor))
            prefix.make_call(call)
        return np.exp(logw - logw.max())
