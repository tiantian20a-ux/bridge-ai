"""Full game state: deal, auction, then card play."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .auction import Auction, Call, Contract, Denomination
from .cards import Card, Hand, Suit
from .deal import Deal, Seat


class Phase(Enum):
    AUCTION = "auction"
    PLAY = "play"
    DONE = "done"


class IllegalPlay(ValueError):
    pass


@dataclass
class Trick:
    leader: Seat
    cards: list[Card] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.cards) == 4

    @property
    def led_suit(self) -> Suit | None:
        return self.cards[0].suit if self.cards else None

    def winner(self, trump: Suit | None) -> Seat:
        if not self.is_complete:
            raise ValueError("trick is not complete")
        led = self.cards[0].suit

        def strength(c: Card) -> int:
            if trump is not None and c.suit == trump:
                return 100 + c.rank
            if c.suit == led:
                return c.rank
            return -1

        best = max(range(4), key=lambda k: strength(self.cards[k]))
        return self.leader.next(best)


def trump_suit(denomination: Denomination) -> Suit | None:
    return None if denomination is Denomination.NOTRUMP else Suit(int(denomination))


@dataclass
class GameState:
    deal: Deal
    auction: Auction = field(init=False)
    remaining: list[Hand] = field(init=False)
    tricks: list[Trick] = field(default_factory=list, init=False)
    contract: Contract | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.auction = Auction(self.deal.dealer)
        self.remaining = list(self.deal.hands)

    @property
    def phase(self) -> Phase:
        if not self.auction.is_over:
            return Phase.AUCTION
        if self.contract is None or len(self.tricks) == 13 and self.tricks[-1].is_complete:
            return Phase.DONE
        return Phase.PLAY

    @property
    def declarer(self) -> Seat | None:
        return self.contract.declarer if self.contract else None

    @property
    def dummy(self) -> Seat | None:
        return self.contract.declarer.partner if self.contract else None

    @property
    def trump(self) -> Suit | None:
        return trump_suit(self.contract.denomination) if self.contract else None

    @property
    def to_act(self) -> Seat:
        """Seat whose turn it is. In play, dummy's cards are chosen by declarer but this returns dummy's seat."""
        if self.phase is Phase.AUCTION:
            return self.auction.to_act
        if self.phase is Phase.DONE:
            raise ValueError("game is over")
        trick = self.tricks[-1]
        return trick.leader.next(len(trick.cards))

    # --- auction ---------------------------------------------------------

    def make_call(self, call: Call) -> None:
        if self.phase is not Phase.AUCTION:
            raise ValueError("auction is over")
        self.auction.make_call(call)
        if self.auction.is_over:
            self.contract = self.auction.contract()
            if self.contract is not None:
                self.tricks.append(Trick(self.contract.declarer.next()))

    # --- play ------------------------------------------------------------

    def legal_plays(self) -> list[Card]:
        if self.phase is not Phase.PLAY:
            return []
        hand = self.remaining[self.to_act]
        led = self.tricks[-1].led_suit
        if led is not None:
            follow = hand.cards_in(led)
            if follow:
                return follow
        return list(hand)

    def play(self, card: Card) -> None:
        if self.phase is not Phase.PLAY:
            raise IllegalPlay("not in the play phase")
        if card not in self.legal_plays():
            raise IllegalPlay(f"{card} is not a legal play for {self.to_act.symbol}")
        seat = self.to_act
        self.remaining[seat] = self.remaining[seat].remove(card)
        trick = self.tricks[-1]
        trick.cards.append(card)
        if trick.is_complete and len(self.tricks) < 13:
            self.tricks.append(Trick(trick.winner(self.trump)))

    def tricks_won(self, seat: Seat) -> int:
        """Completed tricks won by the side of ``seat``."""
        return sum(
            1 for t in self.tricks if t.is_complete and t.winner(self.trump).same_side(seat)
        )

    def declarer_tricks(self) -> int:
        if self.contract is None:
            raise ValueError("no contract")
        return self.tricks_won(self.contract.declarer)
