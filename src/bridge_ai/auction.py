"""Calls, the auction sequence and the resulting contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Union

from .deal import Seat


class Denomination(IntEnum):
    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3
    NOTRUMP = 4

    @property
    def symbol(self) -> str:
        return ["C", "D", "H", "S", "NT"][self]

    @classmethod
    def from_symbol(cls, s: str) -> Denomination:
        s = s.upper()
        if s in ("N", "NT"):
            return cls.NOTRUMP
        return cls("CDHS".index(s))


@dataclass(frozen=True, order=True)
class Bid:
    level: int
    denomination: Denomination

    def __post_init__(self) -> None:
        if not 1 <= self.level <= 7:
            raise ValueError(f"bid level must be 1..7, got {self.level}")

    @property
    def index(self) -> int:
        """0..34, increasing with rank (1C=0 ... 7NT=34)."""
        return (self.level - 1) * 5 + int(self.denomination)

    def __str__(self) -> str:
        return f"{self.level}{self.denomination.symbol}"


@dataclass(frozen=True)
class _Simple:
    name: str

    def __str__(self) -> str:
        return self.name


Pass = _Simple("P")
Double = _Simple("X")
Redouble = _Simple("XX")

Call = Union[Bid, _Simple]

# Index layout for model action spaces: 0=Pass, 1=X, 2=XX, 3..37 = bids.
NUM_CALLS = 38


def call_index(call: Call) -> int:
    if call is Pass:
        return 0
    if call is Double:
        return 1
    if call is Redouble:
        return 2
    assert isinstance(call, Bid)
    return 3 + call.index


def call_from_index(i: int) -> Call:
    if i == 0:
        return Pass
    if i == 1:
        return Double
    if i == 2:
        return Redouble
    if 3 <= i < NUM_CALLS:
        level, denom = divmod(i - 3, 5)
        return Bid(level + 1, Denomination(denom))
    raise ValueError(f"call index out of range: {i}")


def parse_call(text: str) -> Call:
    t = text.strip().upper()
    if t in ("P", "PASS"):
        return Pass
    if t in ("X", "DBL", "D"):
        return Double
    if t in ("XX", "RDBL", "R"):
        return Redouble
    if len(t) < 2 or not t[0].isdigit():
        raise ValueError(f"bad call: {text!r}")
    return Bid(int(t[0]), Denomination.from_symbol(t[1:]))


@dataclass(frozen=True)
class Contract:
    level: int
    denomination: Denomination
    declarer: Seat
    doubled: int = 0  # 0, 1 (X) or 2 (XX)

    def __str__(self) -> str:
        return f"{self.level}{self.denomination.symbol}{'X' * self.doubled} by {self.declarer.symbol}"


class IllegalCall(ValueError):
    pass


@dataclass
class Auction:
    dealer: Seat
    calls: list[Call] = field(default_factory=list)

    @classmethod
    def from_calls(cls, dealer: Seat, calls: list[Call] | str) -> Auction:
        if isinstance(calls, str):
            calls = [parse_call(c) for c in calls.split()]
        a = cls(dealer)
        for c in calls:
            a.make_call(c)
        return a

    @property
    def to_act(self) -> Seat:
        return self.dealer.next(len(self.calls))

    def _last_bid(self) -> tuple[int, Bid] | None:
        for i in range(len(self.calls) - 1, -1, -1):
            c = self.calls[i]
            if isinstance(c, Bid):
                return i, c
        return None

    def _last_non_pass(self) -> tuple[int, Call] | None:
        for i in range(len(self.calls) - 1, -1, -1):
            if self.calls[i] is not Pass:
                return i, self.calls[i]
        return None

    def _seat_of(self, i: int) -> Seat:
        return self.dealer.next(i)

    @property
    def is_over(self) -> bool:
        if self._last_bid() is None:
            return len(self.calls) >= 4
        return all(c is Pass for c in self.calls[-3:])

    @property
    def is_passed_out(self) -> bool:
        return self.is_over and self._last_bid() is None

    def is_legal(self, call: Call) -> bool:
        if self.is_over:
            return False
        if call is Pass:
            return True
        if isinstance(call, Bid):
            last = self._last_bid()
            return last is None or call.index > last[1].index
        last = self._last_non_pass()
        if last is None:
            return False
        i, prev = last
        by_opponent = not self._seat_of(i).same_side(self.to_act)
        if call is Double:
            return isinstance(prev, Bid) and by_opponent
        if call is Redouble:
            return prev is Double and by_opponent
        return False

    def legal_calls(self) -> list[Call]:
        return [c for c in map(call_from_index, range(NUM_CALLS)) if self.is_legal(c)]

    def make_call(self, call: Call) -> None:
        if not self.is_legal(call):
            raise IllegalCall(f"{call} is not legal for {self.to_act.symbol} after {self}")
        self.calls.append(call)

    def contract(self) -> Contract | None:
        """The final contract, or None if the auction is unfinished or passed out."""
        if not self.is_over or self.is_passed_out:
            return None
        i, bid = self._last_bid()  # type: ignore[misc]
        side = self._seat_of(i)
        declarer = next(
            self._seat_of(j)
            for j, c in enumerate(self.calls)
            if isinstance(c, Bid) and c.denomination == bid.denomination and self._seat_of(j).same_side(side)
        )
        doubled = 0
        for c in self.calls[i + 1:]:
            if c is Double:
                doubled = 1
            elif c is Redouble:
                doubled = 2
        return Contract(bid.level, bid.denomination, declarer, doubled)

    def __str__(self) -> str:
        return " ".join(str(c) for c in self.calls) or "(empty)"
