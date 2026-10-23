"""Double-dummy solver: a ctypes binding to Bo Haglund and Soren Hein's DDS 2.9.

The shared library is not bundled. Build it with ``scripts/build_dds.sh``,
which puts it in ``third_party/dds/``, or point ``BRIDGE_AI_DDS_LIB`` at an
existing ``libdds.so`` / ``libdds.dylib`` / ``dds.dll``.

DDS orders suits S, H, D, C (strain 4 is notrump) and hands N, E, S, W.
Seats here already match; suits and denominations are converted at the
boundary, so callers only see this package's own types.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .auction import Denomination
from .cards import Card, Hand, Rank, Suit
from .deal import Deal, Seat

MAX_TABLES = 40  # MAXNOOFTABLES in dll.h; CalcAllTables takes at most this many deals per call
MAX_BOARDS = 200  # MAXNOOFBOARDS in dll.h; SolveAllChunksBin takes at most this many positions per call
_RETURN_NO_FAULT = 1


class DDSError(RuntimeError):
    pass


class _DdTableDeal(ctypes.Structure):
    _fields_ = [("cards", ctypes.c_uint * 4 * 4)]


class _DdTableDeals(ctypes.Structure):
    _fields_ = [("noOfTables", ctypes.c_int), ("deals", _DdTableDeal * (MAX_TABLES * 5))]


class _DdTableResults(ctypes.Structure):
    _fields_ = [("resTable", ctypes.c_int * 4 * 5)]


class _DdTablesRes(ctypes.Structure):
    _fields_ = [("noOfBoards", ctypes.c_int), ("results", _DdTableResults * (MAX_TABLES * 5))]


class _ParResults(ctypes.Structure):
    _fields_ = [("parScore", ctypes.c_char * 16 * 2), ("parContractsString", ctypes.c_char * 128 * 2)]


class _AllParResults(ctypes.Structure):
    _fields_ = [("presults", _ParResults * MAX_TABLES)]


class _DealBin(ctypes.Structure):
    _fields_ = [
        ("trump", ctypes.c_int),
        ("first", ctypes.c_int),
        ("currentTrickSuit", ctypes.c_int * 3),
        ("currentTrickRank", ctypes.c_int * 3),
        ("remainCards", ctypes.c_uint * 4 * 4),
    ]


class _FutureTricks(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.c_int),
        ("cards", ctypes.c_int),
        ("suit", ctypes.c_int * 13),
        ("rank", ctypes.c_int * 13),
        ("equals", ctypes.c_int * 13),
        ("score", ctypes.c_int * 13),
    ]


class _Boards(ctypes.Structure):
    _fields_ = [
        ("noOfBoards", ctypes.c_int),
        ("deals", _DealBin * MAX_BOARDS),
        ("target", ctypes.c_int * MAX_BOARDS),
        ("solutions", ctypes.c_int * MAX_BOARDS),
        ("mode", ctypes.c_int * MAX_BOARDS),
    ]


class _SolvedBoards(ctypes.Structure):
    _fields_ = [("noOfBoards", ctypes.c_int), ("solvedBoard", _FutureTricks * MAX_BOARDS)]


_lib: ctypes.CDLL | None = None
_lock = threading.Lock()


def _candidate_paths() -> list[Path]:
    env = os.environ.get("BRIDGE_AI_DDS_LIB")
    if env:
        return [Path(env)]
    name = {"darwin": "libdds.dylib", "win32": "dds.dll"}.get(sys.platform, "libdds.so")
    root = Path(__file__).resolve().parents[2]
    return [root / "third_party" / "dds" / name, Path(name)]


def load() -> ctypes.CDLL:
    """Load the DDS library once; raises DDSError if it cannot be found."""
    global _lib
    with _lock:
        if _lib is not None:
            return _lib
        errors = []
        for path in _candidate_paths():
            try:
                lib = ctypes.CDLL(str(path))
                break
            except OSError as e:
                errors.append(f"{path}: {e}")
        else:
            raise DDSError("DDS library not found; run scripts/build_dds.sh or set BRIDGE_AI_DDS_LIB\n"
                           + "\n".join(errors))
        lib.SetMaxThreads.argtypes = [ctypes.c_int]
        lib.SetMaxThreads.restype = None
        lib.CalcDDtable.argtypes = [_DdTableDeal, ctypes.POINTER(_DdTableResults)]
        lib.CalcDDtable.restype = ctypes.c_int
        lib.CalcAllTables.argtypes = [ctypes.POINTER(_DdTableDeals), ctypes.c_int, ctypes.c_int * 5,
                                      ctypes.POINTER(_DdTablesRes), ctypes.POINTER(_AllParResults)]
        lib.CalcAllTables.restype = ctypes.c_int
        lib.SolveBoard.argtypes = [_DealBin, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   ctypes.POINTER(_FutureTricks), ctypes.c_int]
        lib.SolveBoard.restype = ctypes.c_int
        lib.SolveAllChunksBin.argtypes = [ctypes.POINTER(_Boards), ctypes.POINTER(_SolvedBoards), ctypes.c_int]
        lib.SolveAllChunksBin.restype = ctypes.c_int
        lib.ErrorMessage.argtypes = [ctypes.c_int, ctypes.c_char_p]
        lib.ErrorMessage.restype = None
        lib.SetMaxThreads(0)  # auto-configure; required on Linux and macOS
        _lib = lib
        return lib


def available() -> bool:
    try:
        load()
    except DDSError:
        return False
    return True


def _check(lib: ctypes.CDLL, code: int) -> None:
    if code != _RETURN_NO_FAULT:
        buf = ctypes.create_string_buffer(80)
        lib.ErrorMessage(code, buf)
        raise DDSError(f"DDS error {code}: {buf.value.decode()}")


def _dds_suit(suit: Suit) -> int:
    return 3 - int(suit)


def _dds_strain(denomination: Denomination) -> int:
    return 4 if denomination is Denomination.NOTRUMP else 3 - int(denomination)


def _holdings(hands: Sequence[Hand], out) -> None:
    """Fill a [hand][suit] uint array; DDS puts the deuce at bit 2 and the ace at bit 14."""
    for seat in range(4):
        mask = hands[seat].mask
        for suit in Suit:
            out[seat][_dds_suit(suit)] = (mask >> (int(suit) * 13) & 0x1FFF) << 2


DDTable = dict[tuple[Denomination, Seat], int]


def _table(res: _DdTableResults) -> DDTable:
    return {(d, s): res.resTable[_dds_strain(d)][s] for d in Denomination for s in Seat}


def dd_table(deal: Deal) -> DDTable:
    """Tricks declarer takes double dummy, for every denomination and declarer."""
    lib = load()
    td = _DdTableDeal()
    _holdings(deal.hands, td.cards)
    res = _DdTableResults()
    _check(lib, lib.CalcDDtable(td, ctypes.byref(res)))
    return _table(res)


def dd_tables(deals: Iterable[Deal]) -> list[DDTable]:
    """Like dd_table for many deals, solved in parallel by DDS."""
    lib = load()
    deals = list(deals)
    out: list[DDTable] = []
    no_filter = (ctypes.c_int * 5)()
    for start in range(0, len(deals), MAX_TABLES):
        chunk = deals[start:start + MAX_TABLES]
        req = _DdTableDeals()
        req.noOfTables = len(chunk)
        for i, d in enumerate(chunk):
            _holdings(d.hands, req.deals[i].cards)
        res = _DdTablesRes()
        par = _AllParResults()
        _check(lib, lib.CalcAllTables(ctypes.byref(req), -1, no_filter, ctypes.byref(res), ctypes.byref(par)))
        out.extend(_table(res.results[i]) for i in range(len(chunk)))
    return out


@dataclass(frozen=True)
class Position:
    """A card-play position for the solver.

    ``remaining`` holds each seat's unplayed cards (indexed by Seat), not
    counting cards already on the table in the current ``trick``, which
    ``leader`` started.
    """

    remaining: Sequence[Hand]
    trump: Denomination
    leader: Seat
    trick: Sequence[Card] = ()


def _fill_deal(dl: _DealBin, pos: Position) -> None:
    if len(pos.trick) > 3:
        raise ValueError("the current trick has at most three cards before the next play")
    dl.trump = _dds_strain(pos.trump)
    dl.first = int(pos.leader)
    for i in range(3):
        if i < len(pos.trick):
            dl.currentTrickSuit[i] = _dds_suit(pos.trick[i].suit)
            dl.currentTrickRank[i] = int(pos.trick[i].rank) + 2
        else:
            dl.currentTrickSuit[i] = 0
            dl.currentTrickRank[i] = 0
    _holdings(pos.remaining, dl.remainCards)


def _card_values(fut: _FutureTricks) -> dict[Card, int]:
    result: dict[Card, int] = {}
    for i in range(fut.cards):
        suit = Suit(3 - fut.suit[i])
        score = fut.score[i]
        result[Card.of(suit, Rank(fut.rank[i] - 2))] = score
        eq = fut.equals[i]  # lower cards of the same suit that are equivalent
        for r in range(2, 15):
            if eq >> r & 1:
                result[Card.of(suit, Rank(r - 2))] = score
    return result


def solve_position(remaining: Sequence[Hand], trump: Denomination, leader: Seat,
                   trick: Sequence[Card] = (), thread_index: int = 0) -> dict[Card, int]:
    """Double-dummy value of every legal card for the player to act.

    See Position for the arguments. Returns, for each legal card, the number
    of tricks the side to play takes from the remaining tricks, the current
    one included.
    """
    lib = load()
    dl = _DealBin()
    _fill_deal(dl, Position(remaining, trump, leader, trick))
    fut = _FutureTricks()
    # target -1 / solutions 3: score every card; mode 1: search even with one legal card
    _check(lib, lib.SolveBoard(dl, -1, 3, 1, ctypes.byref(fut), thread_index))
    return _card_values(fut)


def solve_positions(positions: Iterable[Position]) -> list[dict[Card, int]]:
    """Like solve_position for many positions, solved in parallel by DDS."""
    lib = load()
    positions = list(positions)
    out: list[dict[Card, int]] = []
    for start in range(0, len(positions), MAX_BOARDS):
        chunk = positions[start:start + MAX_BOARDS]
        req = _Boards()
        req.noOfBoards = len(chunk)
        for i, pos in enumerate(chunk):
            _fill_deal(req.deals[i], pos)
            req.target[i] = -1
            req.solutions[i] = 3
            req.mode[i] = 1
        res = _SolvedBoards()
        _check(lib, lib.SolveAllChunksBin(ctypes.byref(req), ctypes.byref(res), 1))
        out.extend(_card_values(res.solvedBoard[i]) for i in range(len(chunk)))
    return out
