"""Card-play agents and a loop that plays a board out.

``PIMCPlayer`` is perfect-information Monte Carlo: sample layouts of the
unseen cards consistent with the play (and the auction), solve each one
double dummy, and play the card that does best across the samples.
The others are baselines for evaluating it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping, Protocol

from . import dds
from .cards import Card
from .deal import Seat
from .inference import constraints_from_auction
from .sampling import HandConstraint, Sampler, View
from .state import GameState, Phase


class Player(Protocol):
    def choose(self, state: GameState) -> Card:
        """The card to play for ``state.to_act`` (for dummy, as its declarer)."""
        ...


def controller(state: GameState) -> Seat:
    """The seat that chooses the next card: the declarer plays dummy's cards."""
    seat = state.to_act
    return state.declarer if seat == state.dummy else seat  # type: ignore[return-value]


def _position(state: GameState, remaining) -> dds.Position:
    trick = state.tricks[-1]
    return dds.Position(remaining, state.contract.denomination, trick.leader, trick.cards)  # type: ignore[union-attr]


def tricks_needed(state: GameState, seat: Seat) -> int:
    """Total tricks the side of ``seat`` needs: to make the contract, or to beat it."""
    target = state.contract.level + 6  # type: ignore[union-attr]
    return target if seat.same_side(state.declarer) else 14 - target  # type: ignore[arg-type]


class RandomPlayer:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def choose(self, state: GameState) -> Card:
        return self.rng.choice(state.legal_plays())


class SimplePlayer:
    """Rules of thumb with no lookahead.

    Lead the top of a sequence or the fourth best of the longest suit;
    second hand low; otherwise win as cheaply as possible unless partner is
    already winning, and play the lowest card when not winning.
    """

    def choose(self, state: GameState) -> Card:
        legal = state.legal_plays()
        trick = state.tricks[-1]
        trump = state.trump
        if not trick.cards:
            longest = max(legal, key=lambda c: (sum(1 for x in legal if x.suit == c.suit), c.suit)).suit
            suit = sorted((c for c in legal if c.suit == longest), key=lambda c: c.rank, reverse=True)
            if len(suit) >= 2 and suit[0].rank >= 10 and suit[0].rank - suit[1].rank == 1:
                return suit[0]
            return suit[min(3, len(suit) - 1)]

        def strength(c: Card) -> int:
            if trump is not None and c.suit == trump:
                return 100 + c.rank
            return c.rank if c.suit == trick.cards[0].suit else -1

        best_k = max(range(len(trick.cards)), key=lambda k: strength(trick.cards[k]))
        best = strength(trick.cards[best_k])
        partner_winning = len(trick.cards) - best_k == 2
        lowest = min(legal, key=lambda c: (strength(c) >= 100, c.rank))
        if len(trick.cards) == 1 or partner_winning:
            return lowest
        winners = [c for c in legal if strength(c) > best]
        return min(winners, key=strength) if winners else lowest


class DoubleDummyPlayer:
    """Sees all four hands and plays a card that is best double dummy (the lowest of equals)."""

    def choose(self, state: GameState) -> Card:
        values = dds.solve_positions([_position(state, state.remaining)])[0]
        top = max(values.values())
        return min((c for c, v in values.items() if v == top), key=lambda c: (c.rank, c.suit))


@dataclass
class Decision:
    card: Card
    make: float  # share of samples where the side reaches its target
    tricks: float  # mean tricks the side takes from here on
    samples: int


@dataclass
class PIMCPlayer:
    """Perfect-information Monte Carlo.

    For each legal card, average over ``samples`` layouts whether the side
    reaches its target (making or beating the contract) and how many
    tricks it takes; play the card with the best chance of the target,
    then the most tricks, then the lowest rank.
    """

    samples: int = 64
    use_auction: bool = True
    rng: random.Random = field(default_factory=random.Random)
    constraints: Mapping[Seat, HandConstraint] | None = None  # overrides use_auction when set
    last: Decision | None = field(default=None, init=False)

    def _constraints(self, state: GameState) -> dict[Seat, HandConstraint]:
        if self.constraints is not None:
            return dict(self.constraints)
        return constraints_from_auction(state.auction) if self.use_auction else {}

    def evaluate(self, state: GameState) -> dict[Card, tuple[float, float]]:
        """(chance of reaching the target, mean tricks) for each legal card."""
        seat = controller(state)
        sampler = Sampler(View.from_state(state, seat), self._constraints(state), self.rng)
        layouts = sampler.sample(self.samples)
        results = dds.solve_positions(_position(state, layout) for layout in layouts)
        won = state.tricks_won(state.to_act)
        need = tricks_needed(state, state.to_act)
        out = {}
        for card in state.legal_plays():
            vals = [r[card] for r in results]
            out[card] = (sum(won + v >= need for v in vals) / len(vals), sum(vals) / len(vals))
        return out

    def choose(self, state: GameState) -> Card:
        legal = state.legal_plays()
        if len(legal) == 1:
            self.last = Decision(legal[0], float("nan"), float("nan"), 0)
            return legal[0]
        scores = self.evaluate(state)
        card = max(legal, key=lambda c: (scores[c][0], scores[c][1], -c.rank))
        self.last = Decision(card, *scores[card], self.samples)
        return card


def play_out(state: GameState, players: Mapping[Seat, Player]) -> int:
    """Play the board to the end; returns the declarer's tricks.

    ``players`` is keyed by the seat that chooses: dummy's cards come from
    the declarer's player, so dummy needs no entry.
    """
    while state.phase is Phase.PLAY:
        state.play(players[controller(state)].choose(state))
    return state.declarer_tricks()
