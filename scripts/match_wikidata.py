"""Match creator names against Wikidata and rank by notability.

For each candidate name (from reports/names.csv plus the curated seed list),
searches Wikidata, keeps entities that are human (P31=Q5) with an artist
occupation (P106), and ranks by sitelink count (number of Wikipedia language
editions - a reliable, citable notability proxy).

Lookups are cached in a small sqlite file, so the script is resumable and
re-runs are free. Only cheap API calls are used (wbsearchentities +
wbgetentities); no dumps required.

Usage:
    python scripts/match_wikidata.py [--db PATH] [--max-names 5000] [--min-images 1]
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path

import httpx

from _common import DEFAULT_REPORTS, REPO_ROOT, add_common_arguments, display_form, ensure_dir, name_key, write_csv

API_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = "smithsonian-subsample/0.1 (dataset curation research; httpx)"
HUMAN_QID = "Q5"

ARTIST_OCCUPATIONS = {
    "Q483501": "artist",
    "Q3391743": "visual artist",
    "Q1028181": "painter",
    "Q1281618": "sculptor",
    "Q33231": "photographer",
    "Q11569986": "printmaker",
    "Q644687": "illustrator",
    "Q329439": "engraver",
    "Q10862983": "etcher",
    "Q15296811": "drawer",
    "Q1925963": "graphic artist",
    "Q7541856": "ceramicist",
    "Q1114448": "cartoonist",
    "Q42973": "architect",
    "Q5322166": "designer",
    "Q627325": "graphic designer",
    "Q21550646": "calligrapher",
    "Q17505902": "watercolorist",
}

DEFAULT_SEED_FILE = REPO_ROOT / "scripts" / "data" / "seed_artists.txt"


class LookupCache:
    def __init__(self, path: Path) -> None:
        ensure_dir(path.parent)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS lookups (name_key TEXT PRIMARY KEY, payload TEXT NOT NULL, updated_at INTEGER NOT NULL)"
        )

    def get(self, key: str) -> dict | None:
        row = self._conn.execute("SELECT payload FROM lookups WHERE name_key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, payload: dict) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO lookups (name_key, payload, updated_at) VALUES (?, ?, ?)",
            (key, json.dumps(payload, ensure_ascii=False), int(time.time())),
        )
        self._conn.commit()


def _api_get(client: httpx.Client, params: dict, retries: int = 5) -> dict:
    params = {**params, "format": "json"}
    for attempt in range(retries + 1):
        try:
            response = client.get(API_URL, params=params)
            if response.status_code in (429, 500, 502, 503):
                raise httpx.HTTPStatusError("retryable", request=response.request, response=response)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            if attempt >= retries:
                raise
            time.sleep(min(2.0**attempt, 30.0))
    raise RuntimeError("unreachable")


def _claim_ids(entity: dict, prop: str) -> set[str]:
    ids = set()
    for claim in entity.get("claims", {}).get(prop, []):
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, dict) and value.get("id"):
            ids.add(value["id"])
    return ids


def resolve_name(client: httpx.Client, name: str, sleep: float) -> dict:
    """Return {"qid", "label", "sitelinks", "occupations"} or {"qid": None}."""
    search = _api_get(
        client,
        {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "uselang": "en",
            "type": "item",
            "limit": 7,
        },
    )
    time.sleep(sleep)
    candidate_ids = [hit["id"] for hit in search.get("search", [])]
    if not candidate_ids:
        return {"qid": None}

    entities = _api_get(
        client,
        {
            "action": "wbgetentities",
            "ids": "|".join(candidate_ids),
            "props": "claims|sitelinks|labels",
            "languages": "en",
        },
    )
    time.sleep(sleep)

    best: dict | None = None
    for qid in candidate_ids:
        entity = entities.get("entities", {}).get(qid) or {}
        if HUMAN_QID not in _claim_ids(entity, "P31"):
            continue
        occupations = _claim_ids(entity, "P106") & set(ARTIST_OCCUPATIONS)
        if not occupations:
            continue
        sitelinks = len(entity.get("sitelinks", {}))
        candidate = {
            "qid": qid,
            "label": entity.get("labels", {}).get("en", {}).get("value", ""),
            "sitelinks": sitelinks,
            "occupations": sorted(ARTIST_OCCUPATIONS[o] for o in occupations),
        }
        if best is None or sitelinks > best["sitelinks"]:
            best = candidate
    return best or {"qid": None}


def load_seed_names(path: Path) -> dict[str, str]:
    seeds: dict[str, str] = {}
    if not path.exists():
        return seeds
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        key = name_key(text)
        if key:
            seeds.setdefault(key, text)
    return seeds


def load_names_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--names-csv", type=Path, default=DEFAULT_REPORTS / "names.csv")
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--max-names", type=int, default=5000, help="Look up at most N database names (by image count).")
    parser.add_argument("--min-images", type=int, default=1, help="Skip names attached to fewer images.")
    parser.add_argument("--min-records", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.15, help="Delay between API calls.")
    parser.add_argument("--exclude-roles", default="Sitter", help="Comma-separated top_role values to skip (still matched if seeded).")
    args = parser.parse_args()

    if not args.names_csv.exists():
        raise SystemExit(f"{args.names_csv} not found - run extract_names.py first")

    reports = ensure_dir(args.reports_dir)
    cache = LookupCache(reports / "wikidata_cache.sqlite3")
    seeds = load_seed_names(args.seed_file)
    names = load_names_csv(args.names_csv)
    excluded_roles = {role.strip().casefold() for role in args.exclude_roles.split(",") if role.strip()}

    by_key: dict[str, dict] = {}
    for row in names:
        by_key.setdefault(row["name_key"], row)

    # Queue: seeds first (always), then DB names by image count.
    queue: list[tuple[str, str]] = [(key, name) for key, name in seeds.items()]
    db_candidates = 0
    for row in names:
        key = row["name_key"]
        if key in seeds:
            continue
        if int(row["image_count"]) < args.min_images or int(row["record_count"]) < args.min_records:
            continue
        if row["top_role"].casefold() in excluded_roles:
            continue
        if db_candidates >= args.max_names:
            break
        queue.append((key, row["name"]))
        db_candidates += 1

    pending = [(key, name) for key, name in queue if cache.get(key) is None]
    print(f"names queued: {len(queue):,} (seeds: {len(seeds):,}), uncached: {len(pending):,}")

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30.0) as client:
        for index, (key, name) in enumerate(pending, 1):
            try:
                result = resolve_name(client, name, args.sleep)
            except httpx.HTTPError as error:
                print(f"  ! {name}: {error} (will retry next run)")
                continue
            cache.put(key, result)
            if index % 50 == 0 or index == len(pending):
                print(f"  {index}/{len(pending)} looked up")

    rows = []
    for key, name in queue:
        result = cache.get(key)
        if not result or not result.get("qid"):
            continue
        db_row = by_key.get(key, {})
        rows.append(
            [
                db_row.get("name") or display_form(name),
                key,
                result["qid"],
                result.get("label", ""),
                result.get("sitelinks", 0),
                ";".join(result.get("occupations", [])),
                1 if key in seeds else 0,
                int(db_row.get("record_count", 0) or 0),
                int(db_row.get("image_count", 0) or 0),
            ]
        )
    rows.sort(key=lambda row: (-row[6], -row[4], row[0]))

    path = write_csv(
        reports / "artist_rankings.csv",
        ["name", "name_key", "qid", "wikidata_label", "sitelinks", "occupations", "is_seed", "record_count", "image_count"],
        rows,
    )
    matched_db = sum(1 for row in rows if row[7] > 0)
    print(f"ranked artists: {len(rows):,} ({matched_db:,} present in database)")
    print(f"output: {path}")


if __name__ == "__main__":
    main()
