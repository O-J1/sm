"""Shared helpers for the Smithsonian analysis scripts.

All scripts open the scrape database read-only and never mutate it.
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "smithsonian_scraper.sqlite3"
DEFAULT_REPORTS = REPO_ROOT / "reports"

# Tier policy (user decision): main image of a record is capped at 8,388,608
# pixels, additional views at 4,194,304 pixels, both encoded as JXL q=97.
MAIN_MAX_PIXELS = 8_388_608
SECONDARY_MAX_PIXELS = 4_194_304
DEFAULT_JXL_QUALITY = 97

MP_BUCKETS = ("<1MP", "1-2MP", "2-4MP", "4-8MP", "8-16MP", "16-32MP", "32-64MP", "64MP+")

# Fallback bytes-per-pixel for JXL q=97 photographic content, used until
# calibrate_bpp.py produces reports/calibration.json with measured values.
FALLBACK_NATIVE_BPP = 0.30
FALLBACK_DOWNSCALED_BPP = 0.40

_PAREN_RE = re.compile(r"\([^)]*\)")
_YEAR_SUFFIX_RE = re.compile(
    r"[,;]?\s*(?:b\.|d\.|ca\.|circa|active|fl\.)?\s*\d{3,4}\s*[-\u2013]?\s*(?:\d{3,4})?\s*$"
)
_NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def name_key(raw: str) -> str:
    """Canonical matching key for a personal name.

    Handles "Last, First" vs "First Last" forms, trailing life dates,
    parentheticals and punctuation by keying on the sorted casefolded tokens.
    """
    text = _PAREN_RE.sub(" ", raw)
    text = _YEAR_SUFFIX_RE.sub(" ", text)
    text = _NON_WORD_RE.sub(" ", text.casefold())
    tokens = sorted(token for token in _WS_RE.split(text) if token and not token.isdigit())
    return " ".join(tokens)


def display_form(raw: str) -> str:
    """Best-effort natural form: "Homer, Winslow, 1836-1910" -> "Winslow Homer"."""
    text = _PAREN_RE.sub(" ", raw)
    text = _YEAR_SUFFIX_RE.sub("", text).strip().strip(",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) >= 2:
        return _WS_RE.sub(" ", f"{parts[1]} {parts[0]}").strip()
    return _WS_RE.sub(" ", text).strip()


def open_db(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise SystemExit(f"database not found: {path}")
    conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to the scrape sqlite database.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS, help="Directory for outputs.")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def histogram_median(hist: dict[int, int]) -> float:
    """Weighted median of a {value: frequency} histogram."""
    total = sum(hist.values())
    if not total:
        return 0.0
    seen = 0
    for value in sorted(hist):
        seen += hist[value]
        if seen * 2 >= total:
            return float(value)
    return 0.0


def mp_bucket(pixels: int) -> str:
    mp = pixels / 1_000_000
    if mp < 1:
        return MP_BUCKETS[0]
    if mp < 2:
        return MP_BUCKETS[1]
    if mp < 4:
        return MP_BUCKETS[2]
    if mp < 8:
        return MP_BUCKETS[3]
    if mp < 16:
        return MP_BUCKETS[4]
    if mp < 32:
        return MP_BUCKETS[5]
    if mp < 64:
        return MP_BUCKETS[6]
    return MP_BUCKETS[7]


class GroupedCursor:
    """Keyed lookup over a cursor sorted by its first column.

    Lookups must be made with ascending keys; rows for skipped keys are
    discarded. Enables merge-joins over multiple record_id-ordered cursors
    without holding whole tables in memory.
    """

    def __init__(self, cursor: sqlite3.Cursor) -> None:
        self._groups = self._grouped(cursor)
        self._done = False
        self._current_key = None
        self._current_rows: list[sqlite3.Row] = []
        self._advance()

    @staticmethod
    def _grouped(cursor: sqlite3.Cursor):
        key = None
        rows: list[sqlite3.Row] = []
        for row in cursor:
            row_key = row[0]
            if row_key != key and key is not None:
                yield key, rows
                rows = []
            key = row_key
            rows.append(row)
        if key is not None:
            yield key, rows

    def _advance(self) -> None:
        try:
            self._current_key, self._current_rows = next(self._groups)
        except StopIteration:
            self._done = True
            self._current_key = None
            self._current_rows = []

    def rows_for(self, key) -> list[sqlite3.Row]:
        while not self._done and self._current_key < key:
            self._advance()
        if not self._done and self._current_key == key:
            return self._current_rows
        return []


def load_bpp(calibration_path: Path) -> tuple[float, float]:
    """Return (native_bpp, downscaled_bpp) from calibration.json or fallbacks."""
    if calibration_path.exists():
        import json

        data = json.loads(calibration_path.read_text(encoding="utf-8"))
        native = (data.get("native_bpp") or {}).get("median")
        caps = [
            value
            for key in ("main_cap_bpp", "secondary_cap_bpp")
            for value in [(data.get(key) or {}).get("median")]
            if value
        ]
        downscaled = sum(caps) / len(caps) if caps else None
        if native:
            return float(native), float(downscaled or native * 1.3)
    return FALLBACK_NATIVE_BPP, FALLBACK_DOWNSCALED_BPP


def load_rank_map(rankings_csv: Path) -> dict[str, int]:
    """Map name_key -> rank (0-based) from artist_rankings.csv (already sorted:
    seeds first, then by sitelinks descending)."""
    if not rankings_csv.exists():
        raise SystemExit(f"{rankings_csv} not found - run match_wikidata.py first")
    ranks: dict[str, int] = {}
    with rankings_csv.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            ranks.setdefault(row["name_key"], index)
    return ranks
