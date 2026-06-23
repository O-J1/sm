from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class SmithsonianRecord:
    id: str
    title: str
    unit_code: str
    linked_id: str
    type: str
    url: str
    content: JsonObject
    hash: str
    doc_signature: str
    timestamp: int | None
    last_time_updated: int | None
    status: int | None
    version: str
    public_search: bool | None
    extensions: JsonObject
    raw: JsonObject


@dataclass(frozen=True)
class FreetextEntry:
    record_id: str
    unit_code: str
    category: str
    label: str
    content: str
    position: int


@dataclass(frozen=True)
class MediaAsset:
    record_id: str
    unit_code: str
    record_hash: str
    url: str
    kind: str = "media"
    media_type: str = ""
    thumbnail: str = ""
    caption: str = ""
    preferred_citation: str = ""
    usage_access: str = ""
    usage_text: str = ""
    usage_codes: tuple[str, ...] = ()
    usage_flag: str = ""
    guid: str = ""
    media_id: str = ""
    ids_id: str = ""
    alt_text: str = ""
    extended_description: str = ""
    resource_label: str = ""
    resource_width: int | None = None
    resource_height: int | None = None
    resource_dimensions: str = ""
    parent_media_url: str = ""
    screen_url: str = ""
    thumbnail_url: str = ""
    downloadable: bool = True

    @property
    def key(self) -> str:
        source = "|".join([self.record_id, self.kind, self.guid, self.media_id, self.resource_label, self.url])
        return hashlib.sha1(source.encode("utf-8")).hexdigest()


def stable_file_stem(value: str, fallback: str = "asset") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned[:160] or fallback


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()