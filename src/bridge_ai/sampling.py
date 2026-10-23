"""Deal sampling: random layouts of the unseen cards consistent with what one seat knows.

A seat in the play sees its own cards, dummy once the opening lead is made,
and every card played so far. A player who failed to follow suit is void
in it. On top of that, constraints on the original thirteen-card hands
(from the auction, say) narrow the layouts further.

Suit lengths and voids are sampled exactly uniformly: a dynamic program
over the four suits counts the layouts for every split of each suit among
the hidden seats. High-card point ranges are applied by rejection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import comb

from .cards import Hand, Suit
from .deal import Seat
from .state import GameState


@dataclass(frozen=True)
class HandConstraint:
    """Ranges (inclusive) for one seat's original thirteen cards."""

    hcp: tuple[int, int] = (0, 37)
    lengths: tuple[tuple[int, int], ...] = ((0, 13),) * 4  # indexed by Suit

    def intersect(self, other: HandConstraint) -> HandConstraint:
        def meet(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
            return max(a[0], b[0]), min(a[1], b[1])

        return HandConstraint(meet(self.hcp, other.hcp),
                              tuple(meet(a, b) for a, b in zip(self.lengths, other.lengths)))

    @property
    def is_trivial(self) -> bool:
        return self == HandConstraint()


NO_CONSTRAINT = HandConstraint()


@dataclass(frozen=True)
class View:
    """What ``seat`` knows about the cards during the play."""

    seat: Seat
    remaining: tuple[Hand | None, ...]  # unplayed cards of seen seats, None for hidden ones
    counts: tuple[int, ...]  # number of unplayed cards each seat holds
    played: tuple[Hand, ...]  # cards each seat has played, the current trick included
    voids: tuple[frozenset[Suit], ...]
    unseen: Hand

    @property
    def hidden(self) -> list[Seat]:
        return [s for s in Seat if self.remaining[s] is None]

    @classmethod
    def from_state(cls, state: GameState, seat: Seat) -> View:
        """The view of ``seat``; for dummy's plays pass the declarer, who chooses them."""
        if state.contract is None:
            raise ValueError("no play without a contract")
        seen = {seat}
        if any(t.cards for t in state.tricks):
            seen.add(state.dummy)
        played = [0] * 4
        voids: list[set[Suit]] = [set() for _ in Seat]
        for t in state.tricks:
            for k, card in enumerate(t.cards):
                s = t.leader.next(k)
                played[s] |= 1 << card.index
                if card.suit != t.cards[0].suit:
                    voids[s].add(t.cards[0].suit)
        known = 0
        for s in seen:
            known |= state.remaining[s].mask
        for m in played:
            known |= m
        return cls(
            seat=seat,
            remaining=tuple(state.remaining[s] if s in seen else None for s in Seat),
            counts=tuple(len(state.remaining[s]) for s in Seat),
            played=tuple(Hand(m) for m in played),
            voids=tuple(frozenset(v) for v in voids),
            unseen=Hand(((1 << 52) - 1) & ~known),
        )


def _count_ranges(view: View, constraints: dict[Seat, HandConstraint]) -> list[list[tuple[int, int]]]:
    """For each hidden seat and suit, how many unseen cards of the suit it may hold."""
    out = []
    for s in view.hidden:
        c = constraints.get(s, NO_CONSTRAINT)
        row = []
        for suit in Suit:
            if suit in view.voids[s]:
                row.append((0, 0))
                continue
            done = view.played[s].suit_length(suit)
            lo, hi = c.lengths[suit]
            row.append((max(0, lo - done), max(-1, hi - done)))
        out.append(row)
    return out


class Sampler:
    """Draws layouts of the unseen cards for one view."""

    def __init__(self, view: View, constraints: dict[Seat, HandConstraint] | None = None,
                 rng: random.Random | None = None):
        self.view = view
        self.rng = rng or random.Random()
        self.hidden = view.hidden
        constraints = {s: c for s, c in (constraints or {}).items() if s in self.hidden and not c.is_trivial}
        self.constraints = constraints
        self.relaxed = False
        self._suit_cards = [[c for c in view.unseen if c.suit == suit] for suit in Suit]
        self._ranges = _count_ranges(view, constraints)
        if self._total() == 0:
            # The constraints contradict the play (a misleading auction): keep only the voids.
            self.relaxed = True
            self.constraints = {}
            self._ranges = _count_ranges(view, {})
            if self._total() == 0:
                raise ValueError("no layout of the unseen cards fits the play")

    def _splits(self, suit: int, need: tuple[int, ...]) -> list[tuple[tuple[int, ...], int]]:
        """Ways to split the suit's unseen cards among hidden seats, with their layout counts."""
        n = len(self._suit_cards[suit])
        out = []

        def rec(i: int, left: int, acc: tuple[int, ...], ways: int) -> None:
            if i == len(need) - 1:
                lo, hi = self._ranges[i][suit]
                if lo <= left <= min(hi, need[i]):
                    out.append((acc + (left,), ways))
                return
            lo, hi = self._ranges[i][suit]
            for k in range(lo, min(hi, need[i], left) + 1):
                rec(i + 1, left - k, acc + (k,), ways * comb(left, k))

        rec(0, n, (), 1)
        return out

    def _total(self) -> int:
        self._memo: dict[tuple[int, tuple[int, ...]], int] = {}
        return self._ways(0, tuple(self.view.counts[s] for s in self.hidden))

    def _ways(self, suit: int, need: tuple[int, ...]) -> int:
        """Layouts of suits ``suit``.. given how many cards each hidden seat still needs."""
        if suit == 4:
            return 1 if not any(need) else 0
        key = (suit, need)
        if key in self._memo:
            return self._memo[key]
        total = 0
        for split, ways in self._splits(suit, need):
            rest = tuple(a - b for a, b in zip(need, split))
            total += ways * self._ways(suit + 1, rest)
        self._memo[key] = total
        return total

    def _draw_layout(self) -> list[Hand]:
        hands = [0] * len(self.hidden)
        need = tuple(self.view.counts[s] for s in self.hidden)
        for suit in range(4):
            options = []
            for split, ways in self._splits(suit, need):
                rest = tuple(a - b for a, b in zip(need, split))
                w = ways * self._ways(suit + 1, rest)
                if w:
                    options.append((split, w))
            pick = self.rng.randrange(sum(w for _, w in options))
            for split, w in options:
                if pick < w:
                    break
                pick -= w
            cards = self._suit_cards[suit][:]
            self.rng.shuffle(cards)
            pos = 0
            for i, k in enumerate(split):
                for c in cards[pos:pos + k]:
                    hands[i] |= 1 << c.index
                pos += k
            need = tuple(a - b for a, b in zip(need, split))
        return [Hand(m) for m in hands]

    def _fits_hcp(self, layout: list[Hand]) -> bool:
        for i, s in enumerate(self.hidden):
            c = self.constraints.get(s)
            if c is None:
                continue
            hcp = layout[i].hcp() + self.view.played[s].hcp()
            if not c.hcp[0] <= hcp <= c.hcp[1]:
                return False
        return True

    def sample(self, n: int, max_tries: int | None = None) -> list[list[Hand]]:
        """``n`` layouts, each the unplayed cards of all four seats (indexed by Seat).

        Draws that miss the point-count ranges are rejected. If fewer than
        ``n`` fit within ``max_tries`` draws, the rest ignore the point
        counts and ``relaxed`` is set.
        """
        max_tries = max_tries if max_tries is not None else 100 * n
        out: list[list[Hand]] = []
        tries = 0
        check = any(c.hcp != (0, 37) for c in self.constraints.values())
        while len(out) < n:
            layout = self._draw_layout()
            tries += 1
            if check and tries <= max_tries and not self._fits_hcp(layout):
                continue
            if tries > max_tries:
                self.relaxed = True
            full = list(self.view.remaining)
            for i, s in enumerate(self.hidden):
                full[s] = layout[i]
            out.append(full)  # type: ignore[arg-type]
        return out

