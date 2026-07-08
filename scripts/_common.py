"""Shared helpers for the Smithsonian analysis scripts.

All scripts open the scrape database read-only and never mutate it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "smithsonian_scraper.sqlite3"
DEFAULT_REPORTS = REPO_ROOT / "reports"

# Tier policy (user decisions):
#   full      - records by a top-N ranked artist keep every image at native
#               resolution (JXL q=95).
#   main      - a record's main image (first media item) capped at 8,388,608 px
#               (NMNHBOTANY: 4,194,304 px).
#   secondary - additional views capped at 2,097,152 px.
#   drop      - images below 1MP (low quality), and NMNHBOTANY views beyond the
#               first 10 of a specimen record.
# Derivative outputs: kept non-full images whose native size is at least 4x
# their highres cap also get 256px and 512px longest-edge Lanczos2Sharp
# resizes generated from the original file (pretraining set).
MAIN_MAX_PIXELS = 8_388_608
SECONDARY_MAX_PIXELS = 2_097_152
DEFAULT_JXL_QUALITY = 95
LOW_QUALITY_MIN_PIXELS = 1_000_000
BOTANY_UNIT = "NMNHBOTANY"
BOTANY_MAIN_MAX_PIXELS = 4_194_304
BOTANY_MAX_IMAGES_PER_RECORD = 10
DERIVATIVE_EDGES = (256, 512)
DERIVATIVE_MIN_REDUCTION = 4.0
DEFAULT_DERIVATIVE_FORMAT = "jpg"
DEFAULT_DERIVATIVE_QUALITY = 90

MP_BUCKETS = ("<1MP", "1-2MP", "2-4MP", "4-8MP", "8-16MP", "16-32MP", "32-64MP", "64MP+")

# Fallback bytes-per-pixel for JXL q=95 photographic content, used until
# calibrate_bpp.py produces reports/calibration.json with measured values.
FALLBACK_NATIVE_BPP = 0.30
FALLBACK_DOWNSCALED_BPP = 0.40
# Fallback bytes-per-pixel for small JPEG derivatives (they compress worse
# per pixel than large photos).
FALLBACK_DERIVATIVE_BPP = 0.60


@dataclass(frozen=True)
class ImageDecision:
    """Output decision for one source image.

    tier: full | main | secondary | drop
    target_max_pixels: highres pixel cap (None = keep native resolution)
    derivative_edges: longest-edge sizes for Lanczos2Sharp pretraining outputs
    """

    tier: str
    target_max_pixels: int | None = None
    drop_reason: str | None = None
    derivative_edges: tuple[int, ...] = ()


@dataclass
class SubsamplePolicy:
    quality: int = DEFAULT_JXL_QUALITY
    main_max_pixels: int = MAIN_MAX_PIXELS
    secondary_max_pixels: int = SECONDARY_MAX_PIXELS
    low_quality_min_pixels: int = LOW_QUALITY_MIN_PIXELS
    unit_main_max_pixels: dict[str, int] = field(
        default_factory=lambda: {BOTANY_UNIT: BOTANY_MAIN_MAX_PIXELS}
    )
    unit_max_images: dict[str, int] = field(
        default_factory=lambda: {BOTANY_UNIT: BOTANY_MAX_IMAGES_PER_RECORD}
    )
    derivative_edges: tuple[int, ...] = DERIVATIVE_EDGES
    derivative_min_reduction: float = DERIVATIVE_MIN_REDUCTION
    derivative_format: str = DEFAULT_DERIVATIVE_FORMAT
    derivative_quality: int = DEFAULT_DERIVATIVE_QUALITY

    def classify(self, *, unit_code: str, index: int, pixels: int, is_top_artist: bool) -> ImageDecision:
        """Classify one image of a record. `index` is the record-local media
        position (0 = main image); `pixels` the native (or estimated) size."""
        if pixels < self.low_quality_min_pixels:
            # Applies to top artists too: sub-1MP files are thumbnails/junk.
            return ImageDecision("drop", drop_reason="low_quality")
        if is_top_artist:
            return ImageDecision("full")
        image_cap = self.unit_max_images.get(unit_code)
        if image_cap is not None and index >= image_cap:
            return ImageDecision("drop", drop_reason="unit_image_cap")
        if index == 0:
            tier = "main"
            cap = self.unit_main_max_pixels.get(unit_code, self.main_max_pixels)
        else:
            tier = "secondary"
            cap = self.secondary_max_pixels
        edges = self.derivative_edges if pixels >= cap * self.derivative_min_reduction else ()
        return ImageDecision(tier, target_max_pixels=cap, derivative_edges=tuple(edges))


def derivative_pixels(width: int | None, height: int | None, edge: int) -> int:
    """Pixel count of an aspect-preserving resize to `edge` on the longest side."""
    if width and height:
        scale = edge / max(width, height)
        if scale >= 1.0:
            return width * height
        return max(1, round(width * scale)) * max(1, round(height * scale))
    return int(edge * edge * 0.75)  # assume 4:3 when dimensions are unknown


def add_policy_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("subsampling policy")
    group.add_argument("--quality", type=int, default=DEFAULT_JXL_QUALITY, help="JXL quality for all highres outputs.")
    group.add_argument("--low-quality-min-mp", type=float, default=LOW_QUALITY_MIN_PIXELS / 1e6,
                       help="Drop images below this many megapixels entirely.")
    group.add_argument("--main-cap-mp", type=float, default=MAIN_MAX_PIXELS / 1e6,
                       help="Pixel cap (MP) for main images.")
    group.add_argument("--secondary-cap-mp", type=float, default=SECONDARY_MAX_PIXELS / 1e6,
                       help="Pixel cap (MP) for additional views.")
    group.add_argument("--botany-main-cap-mp", type=float, default=BOTANY_MAIN_MAX_PIXELS / 1e6,
                       help="Pixel cap (MP) for NMNHBOTANY main images.")
    group.add_argument("--botany-max-images", type=int, default=BOTANY_MAX_IMAGES_PER_RECORD,
                       help="Keep at most this many images per NMNHBOTANY record (0 = unlimited).")
    group.add_argument("--derivative-edges", default=",".join(str(edge) for edge in DERIVATIVE_EDGES),
                       help="Comma-separated longest-edge sizes for pretraining derivatives ('' = none).")
    group.add_argument("--derivative-min-reduction", type=float, default=DERIVATIVE_MIN_REDUCTION,
                       help="Generate derivatives only when native pixels >= this multiple of the highres cap.")
    group.add_argument("--derivative-format", default=DEFAULT_DERIVATIVE_FORMAT, choices=("jpg", "png", "jxl"))
    group.add_argument("--derivative-quality", type=int, default=DEFAULT_DERIVATIVE_QUALITY)


def policy_from_args(args: argparse.Namespace) -> SubsamplePolicy:
    edges = tuple(int(part) for part in str(args.derivative_edges).split(",") if part.strip())
    unit_max_images = {BOTANY_UNIT: args.botany_max_images} if args.botany_max_images > 0 else {}
    return SubsamplePolicy(
        quality=args.quality,
        main_max_pixels=int(args.main_cap_mp * 1e6),
        secondary_max_pixels=int(args.secondary_cap_mp * 1e6),
        low_quality_min_pixels=int(args.low_quality_min_mp * 1e6),
        unit_main_max_pixels={BOTANY_UNIT: int(args.botany_main_cap_mp * 1e6)},
        unit_max_images=unit_max_images,
        derivative_edges=edges,
        derivative_min_reduction=args.derivative_min_reduction,
        derivative_format=args.derivative_format,
        derivative_quality=args.derivative_quality,
    )

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
    # Path("") silently becomes "." - catch empty --db (e.g. unset $db) early.
    if str(path) in ("", ".") or path.is_dir():
        raise SystemExit(f"--db must point to a sqlite database file, got: {path!r} (is your $db variable set?)")
    if not path.is_file():
        raise SystemExit(f"database not found: {path}")
    try:
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
    except sqlite3.OperationalError as error:
        # WAL databases and some filesystems reject read-only opens
        # (shm/locking). Fall back to a normal open; scripts only read.
        print(f"warning: read-only open failed ({error}); opening read-write instead")
        conn = sqlite3.connect(path)
        conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
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


def load_calibration(calibration_path: Path) -> dict:
    if calibration_path.exists():
        return json.loads(calibration_path.read_text(encoding="utf-8"))
    return {}


def load_bpp(calibration_path: Path) -> tuple[float, float]:
    """Return (native_bpp, downscaled_bpp) from calibration.json or fallbacks."""
    if calibration_path.exists():
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


def load_derivative_bpp(calibration_path: Path, edges: Sequence[int]) -> dict[int, float]:
    """Measured bytes-per-pixel per derivative edge, with a conservative fallback."""
    measured = (load_calibration(calibration_path).get("derivative_bpp") or {})
    result: dict[int, float] = {}
    for edge in edges:
        stats = measured.get(str(edge)) or {}
        result[edge] = float(stats.get("median") or FALLBACK_DERIVATIVE_BPP)
    return result


def warn_calibration_quality(calibration_path: Path, requested_quality: int) -> None:
    """Print a loud warning when calibration.json was measured at a different quality."""
    calibrated = load_calibration(calibration_path).get("quality")
    if calibrated is None:
        print(f"warning: {calibration_path} missing - using fallback bpp; run calibrate_bpp.py --quality {requested_quality}")
    elif int(calibrated) != requested_quality:
        print(
            f"warning: calibration measured at q={calibrated} but estimating q={requested_quality}; "
            f"re-run calibrate_bpp.py --quality {requested_quality} for trustworthy numbers"
        )
