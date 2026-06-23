from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .models import FreetextEntry, JsonObject, MediaAsset, SmithsonianRecord


class RecordParseError(ValueError):
    pass


def parse_record(raw: JsonObject) -> SmithsonianRecord:
    record_id = _required_string(raw, "id")
    content = raw.get("content")
    if not isinstance(content, dict):
        raise RecordParseError(f"record {record_id} has no JSON content object")

    return SmithsonianRecord(
        id=record_id,
        title=_string(raw.get("title")),
        unit_code=_string(raw.get("unitCode")),
        linked_id=_string(raw.get("linkedId")),
        type=_string(raw.get("type")),
        url=_string(raw.get("url")),
        content=content,
        hash=_string(raw.get("hash")),
        doc_signature=_string(raw.get("docSignature")),
        timestamp=_optional_int(raw.get("timestamp")),
        last_time_updated=_optional_int(raw.get("lastTimeUpdated")),
        status=_optional_int(raw.get("status")),
        version=_string(raw.get("version")),
        public_search=_optional_bool(raw.get("publicSearch")),
        extensions=raw.get("extensions") if isinstance(raw.get("extensions"), dict) else {},
        raw=raw,
    )


def extract_media_assets(record: SmithsonianRecord) -> list[MediaAsset]:
    assets: list[MediaAsset] = []
    seen_urls: set[tuple[str, str]] = set()
    for media in _iter_online_media(record.content):
        for asset in _assets_from_media(record, media):
            identity = (asset.kind, asset.url)
            if asset.url and identity not in seen_urls:
                assets.append(asset)
                seen_urls.add(identity)
    return assets


def extract_freetext_entries(record: SmithsonianRecord) -> list[FreetextEntry]:
    freetext = record.content.get("freetext")
    if not isinstance(freetext, dict):
        return []

    entries: list[FreetextEntry] = []
    position = 0
    for category, raw_items in freetext.items():
        for item in _as_list(raw_items):
            label = ""
            content = ""
            if isinstance(item, dict):
                label = _string(item.get("label"))
                content = _string(item.get("content"))
            else:
                content = _string(item)
            if not content:
                continue
            entries.append(
                FreetextEntry(
                    record_id=record.id,
                    unit_code=record.unit_code,
                    category=_string(category),
                    label=label,
                    content=content,
                    position=position,
                )
            )
            position += 1
    return entries


def _iter_online_media(content: JsonObject) -> Iterable[Any]:
    descriptive = content.get("descriptiveNonRepeating")
    candidates: list[Any] = []
    if isinstance(descriptive, dict):
        candidates.extend(
            descriptive.get(key)
            for key in (
                "online_media",
                "onlineMedia",
                "online_media_group",
                "onlineMediaGroup",
                "media",
            )
        )
    candidates.extend(content.get(key) for key in ("online_media", "onlineMedia", "media"))

    for candidate in candidates:
        if candidate is None:
            continue
        yield from _flatten_media(candidate)


def _flatten_media(value: Any) -> Iterable[Any]:
    if isinstance(value, list):
        for item in value:
            yield from _flatten_media(item)
    elif isinstance(value, dict):
        media = value.get("media")
        if media is not None:
            yield from _flatten_media(media)
        else:
            yield value
    elif isinstance(value, str):
        yield value


def _assets_from_media(record: SmithsonianRecord, media: Any) -> Iterable[MediaAsset]:
    if isinstance(media, str):
        yield MediaAsset(
            record_id=record.id,
            unit_code=record.unit_code,
            record_hash=record.hash,
            url=media,
            kind="media",
        )
        return

    if not isinstance(media, dict):
        return

    media_url = _first_string(media, "content", "url", "uri", "href", "link", "mediaURL", "mediaUrl")
    thumbnail = _first_string(media, "thumbnail", "thumbnailUrl", "thumbnailURL")
    screen_url, thumbnail_url = _resource_reference_urls(media)
    usage = media.get("usage") if isinstance(media.get("usage"), dict) else {}
    usage_codes = usage.get("codes") if isinstance(usage.get("codes"), list) else []

    common = {
        "record_id": record.id,
        "unit_code": record.unit_code,
        "record_hash": record.hash,
        "media_type": _string(media.get("type")),
        "thumbnail": thumbnail,
        "caption": _string(media.get("caption")),
        "preferred_citation": _string(media.get("preferred_citation") or media.get("preferredCitation")),
        "usage_access": _string(usage.get("access")),
        "usage_text": _string(usage.get("text")),
        "usage_codes": tuple(_string(code) for code in usage_codes if _string(code)),
        "usage_flag": _string(media.get("usage_flag") or media.get("usageFlag")),
        "guid": _string(media.get("guid")),
        "media_id": _string(media.get("id")),
        "ids_id": _string(media.get("idsId") or media.get("idsID")),
        "alt_text": _string(media.get("altTextAccessibility") or media.get("altText")),
        "extended_description": _string(media.get("extDescrAccessibility") or media.get("extendedDescription") or media.get("description")),
        "parent_media_url": media_url,
        "screen_url": screen_url,
        "thumbnail_url": thumbnail_url or thumbnail,
    }

    high_resolution_assets = list(_high_resolution_resource_assets(media, common))
    if high_resolution_assets:
        yield from _prefer_tiff_assets(high_resolution_assets)
        return

    if media_url:
        yield MediaAsset(url=media_url, kind="media", **common)


def _high_resolution_resource_assets(media: JsonObject, common: dict[str, Any]) -> Iterable[MediaAsset]:
    resources = media.get("resources")
    if not isinstance(resources, list):
        return
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        label = _string(resource.get("label"))
        url = _string(resource.get("url"))
        if not url or not _is_high_resolution_resource(label, url):
            continue
        yield MediaAsset(
            url=url,
            kind=_resource_kind(label, url),
            resource_label=label,
            resource_width=_optional_int(resource.get("width")),
            resource_height=_optional_int(resource.get("height")),
            resource_dimensions=_string(resource.get("dimensions")),
            downloadable=True,
            **common,
        )


def _prefer_tiff_assets(assets: list[MediaAsset]) -> list[MediaAsset]:
    if any(asset.kind == "highres_tiff" for asset in assets):
        return [asset for asset in assets if asset.kind != "highres_jpeg"]
    return assets


def _resource_reference_urls(media: JsonObject) -> tuple[str, str]:
    screen_url = ""
    thumbnail_url = ""
    resources = media.get("resources")
    if not isinstance(resources, list):
        return screen_url, thumbnail_url
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        label = _string(resource.get("label")).lower()
        url = _string(resource.get("url"))
        if not url:
            continue
        if "screen" in label:
            screen_url = url
        elif "thumbnail" in label or "thumb" in label:
            thumbnail_url = url
    return screen_url, thumbnail_url


def _is_high_resolution_resource(label: str, url: str) -> bool:
    lowered_label = label.lower()
    lowered_url = url.lower()
    return "high-resolution" in lowered_label or lowered_url.endswith((".jpg", ".jpeg", ".tif", ".tiff"))


def _resource_kind(label: str, url: str) -> str:
    lowered = f"{label} {url}".lower()
    if "tif" in lowered:
        return "highres_tiff"
    if "jp" in lowered:
        return "highres_jpeg"
    return "highres"


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _required_string(raw: JsonObject, key: str) -> str:
    value = _string(raw.get(key))
    if not value:
        raise RecordParseError(f"record is missing required field {key!r}")
    return value


def _first_string(raw: JsonObject, *keys: str) -> str:
    for key in keys:
        value = _string(raw.get(key))
        if value:
            return value
    return ""


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None