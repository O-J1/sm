from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .models import JsonObject, SmithsonianRecord, sha1_text


@dataclass(frozen=True)
class TextEntryProjection:
    record_id: str
    unit_code: str
    category: str
    normalized_category: str
    label: str
    normalized_label: str
    content: str
    content_hash: str
    position: int


@dataclass(frozen=True)
class IdentifierProjection:
    record_id: str
    unit_code: str
    identifier_type: str
    identifier_value: str
    source_category: str
    source_label: str
    position: int


@dataclass(frozen=True)
class DateProjection:
    record_id: str
    unit_code: str
    date_text: str
    start_year: int | None
    end_year: int | None
    precision: str
    source_category: str
    source_label: str
    position: int


@dataclass(frozen=True)
class RightProjection:
    record_id: str
    unit_code: str
    rights_text: str
    normalized_rights: str
    source: str
    source_label: str
    position: int


@dataclass(frozen=True)
class FacetProjection:
    record_id: str
    unit_code: str
    facet_type: str
    value: str
    normalized_value: str
    source_path: str
    position: int


@dataclass(frozen=True)
class MediaItemProjection:
    media_key: str
    record_id: str
    unit_code: str
    media_type: str
    guid: str
    media_id: str
    ids_id: str
    caption: str
    preferred_citation: str
    usage_access: str
    usage_text: str
    usage_flag: str
    alt_text: str
    extended_description: str
    parent_url: str
    position: int


@dataclass(frozen=True)
class MediaResourceProjection:
    resource_key: str
    media_key: str
    record_id: str
    unit_code: str
    role: str
    url: str
    label: str
    width: int | None
    height: int | None
    dimensions: str
    downloadable: bool
    preferred_download: bool
    position: int


@dataclass(frozen=True)
class MediaUsageCodeProjection:
    media_key: str
    record_id: str
    unit_code: str
    code: str
    position: int


@dataclass(frozen=True)
class RecordRelationshipProjection:
    record_id: str
    unit_code: str
    target: str
    relation_type: str
    label: str
    source: str
    position: int


@dataclass(frozen=True)
class NormalizedRecordProjection:
    text_entries: tuple[TextEntryProjection, ...]
    identifiers: tuple[IdentifierProjection, ...]
    dates: tuple[DateProjection, ...]
    rights: tuple[RightProjection, ...]
    facets: tuple[FacetProjection, ...]
    media_items: tuple[MediaItemProjection, ...]
    media_resources: tuple[MediaResourceProjection, ...]
    media_usage_codes: tuple[MediaUsageCodeProjection, ...]
    relationships: tuple[RecordRelationshipProjection, ...]


def normalize_record(record: SmithsonianRecord) -> NormalizedRecordProjection:
    text_entries = tuple(_text_entries(record))
    media_items, media_resources, media_usage_codes = _media_projections(record)
    return NormalizedRecordProjection(
        text_entries=text_entries,
        identifiers=tuple(_identifiers(text_entries)),
        dates=tuple(_dates(text_entries)),
        rights=tuple(_rights(record, text_entries)),
        facets=tuple(_facets(record)),
        media_items=tuple(media_items),
        media_resources=tuple(media_resources),
        media_usage_codes=tuple(media_usage_codes),
        relationships=tuple(_relationships(record)),
    )


def _text_entries(record: SmithsonianRecord) -> Iterable[TextEntryProjection]:
    freetext = record.content.get("freetext")
    if not isinstance(freetext, dict):
        return
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
            category_text = _string(category)
            yield TextEntryProjection(
                record_id=record.id,
                unit_code=record.unit_code,
                category=category_text,
                normalized_category=_normalize_token(category_text),
                label=label,
                normalized_label=_normalize_token(label),
                content=content,
                content_hash=sha1_text(content),
                position=position,
            )
            position += 1


def _identifiers(entries: Iterable[TextEntryProjection]) -> Iterable[IdentifierProjection]:
    for entry in entries:
        category = entry.normalized_category
        label = entry.normalized_label
        if category != "identifier" and not any(term in label for term in ("accession", "catalog", "inventory", "id", "number")):
            continue
        yield IdentifierProjection(
            record_id=entry.record_id,
            unit_code=entry.unit_code,
            identifier_type=_identifier_type(entry.label, entry.category),
            identifier_value=entry.content.strip(),
            source_category=entry.category,
            source_label=entry.label,
            position=entry.position,
        )


def _dates(entries: Iterable[TextEntryProjection]) -> Iterable[DateProjection]:
    for entry in entries:
        if entry.normalized_category != "date" and "date" not in entry.normalized_label:
            continue
        start_year, end_year, precision = _parse_years(entry.content)
        yield DateProjection(
            record_id=entry.record_id,
            unit_code=entry.unit_code,
            date_text=entry.content,
            start_year=start_year,
            end_year=end_year,
            precision=precision,
            source_category=entry.category,
            source_label=entry.label,
            position=entry.position,
        )


def _rights(record: SmithsonianRecord, entries: Iterable[TextEntryProjection]) -> Iterable[RightProjection]:
    position = 0
    for entry in entries:
        if "right" not in entry.normalized_category and "right" not in entry.normalized_label:
            continue
        yield RightProjection(
            record_id=entry.record_id,
            unit_code=entry.unit_code,
            rights_text=entry.content,
            normalized_rights=_normalize_rights(entry.content),
            source=entry.category,
            source_label=entry.label,
            position=position,
        )
        position += 1

    metadata_usage = record.content.get("metadata_usage") or record.content.get("metadataUsage")
    if isinstance(metadata_usage, dict):
        access = _string(metadata_usage.get("access"))
        text = _string(metadata_usage.get("text"))
        rights_text = text or access
        if rights_text:
            yield RightProjection(
                record_id=record.id,
                unit_code=record.unit_code,
                rights_text=rights_text,
                normalized_rights=_normalize_rights(access or rights_text),
                source="metadata_usage",
                source_label="",
                position=position,
            )


def _facets(record: SmithsonianRecord) -> Iterable[FacetProjection]:
    indexed = record.content.get("indexedStructured")
    if not isinstance(indexed, dict):
        return
    position = 0
    for field, value in indexed.items():
        for item in _flatten_values(value):
            text = _string(item)
            if not text:
                continue
            field_text = _string(field)
            yield FacetProjection(
                record_id=record.id,
                unit_code=record.unit_code,
                facet_type=_normalize_token(field_text),
                value=text,
                normalized_value=_normalize_value(text),
                source_path=f"indexedStructured.{field_text}",
                position=position,
            )
            position += 1


def _media_projections(
    record: SmithsonianRecord,
) -> tuple[list[MediaItemProjection], list[MediaResourceProjection], list[MediaUsageCodeProjection]]:
    items: list[MediaItemProjection] = []
    resources: list[MediaResourceProjection] = []
    usage_codes: list[MediaUsageCodeProjection] = []
    seen_resources: set[str] = set()

    for media_position, media in enumerate(_iter_online_media(record.content)):
        if isinstance(media, str):
            media = {"content": media}
        if not isinstance(media, dict):
            continue

        parent_url = _first_string(media, "content", "url", "uri", "href", "link", "mediaURL", "mediaUrl")
        media_id = _string(media.get("id"))
        guid = _string(media.get("guid"))
        ids_id = _string(media.get("idsId") or media.get("idsID"))
        media_key = sha1_text("|".join([record.id, guid, media_id, ids_id, parent_url, str(media_position)]))
        usage = media.get("usage") if isinstance(media.get("usage"), dict) else {}
        codes = usage.get("codes") if isinstance(usage.get("codes"), list) else []

        items.append(
            MediaItemProjection(
                media_key=media_key,
                record_id=record.id,
                unit_code=record.unit_code,
                media_type=_string(media.get("type")),
                guid=guid,
                media_id=media_id,
                ids_id=ids_id,
                caption=_string(media.get("caption")),
                preferred_citation=_string(media.get("preferred_citation") or media.get("preferredCitation")),
                usage_access=_string(usage.get("access")),
                usage_text=_string(usage.get("text")),
                usage_flag=_string(media.get("usage_flag") or media.get("usageFlag")),
                alt_text=_string(media.get("altTextAccessibility") or media.get("altText")),
                extended_description=_string(
                    media.get("extDescrAccessibility") or media.get("extendedDescription") or media.get("description")
                ),
                parent_url=parent_url,
                position=media_position,
            )
        )
        for code_position, code in enumerate(codes):
            code_text = _string(code)
            if code_text:
                usage_codes.append(
                    MediaUsageCodeProjection(
                        media_key=media_key,
                        record_id=record.id,
                        unit_code=record.unit_code,
                        code=code_text,
                        position=code_position,
                    )
                )

        candidates = list(_resource_candidates(media, parent_url))
        preferred_key = _preferred_download_key(media_key, candidates)
        for resource_position, candidate in enumerate(candidates):
            url = candidate["url"]
            resource_key = sha1_text("|".join([media_key, candidate["role"], candidate["label"], url]))
            if resource_key in seen_resources:
                continue
            seen_resources.add(resource_key)
            resources.append(
                MediaResourceProjection(
                    resource_key=resource_key,
                    media_key=media_key,
                    record_id=record.id,
                    unit_code=record.unit_code,
                    role=candidate["role"],
                    url=url,
                    label=candidate["label"],
                    width=candidate["width"],
                    height=candidate["height"],
                    dimensions=candidate["dimensions"],
                    downloadable=candidate["downloadable"],
                    preferred_download=resource_key == preferred_key,
                    position=resource_position,
                )
            )
    return items, resources, usage_codes


def _resource_candidates(media: JsonObject, parent_url: str) -> Iterable[dict[str, Any]]:
    if parent_url:
        yield {
            "role": "primary",
            "url": parent_url,
            "label": "",
            "width": None,
            "height": None,
            "dimensions": "",
            "downloadable": True,
        }
    thumbnail = _first_string(media, "thumbnail", "thumbnailUrl", "thumbnailURL")
    if thumbnail and thumbnail != parent_url:
        yield {
            "role": "thumbnail",
            "url": thumbnail,
            "label": "Thumbnail",
            "width": None,
            "height": None,
            "dimensions": "",
            "downloadable": False,
        }
    raw_resources = media.get("resources")
    if not isinstance(raw_resources, list):
        return
    for resource in raw_resources:
        if not isinstance(resource, dict):
            continue
        url = _string(resource.get("url"))
        if not url:
            continue
        label = _string(resource.get("label"))
        role = _resource_role(label, url)
        yield {
            "role": role,
            "url": url,
            "label": label,
            "width": _optional_int(resource.get("width")),
            "height": _optional_int(resource.get("height")),
            "dimensions": _string(resource.get("dimensions")),
            "downloadable": role in {"highres_tiff", "highres_jpeg", "highres", "primary"},
        }


def _preferred_download_key(media_key: str, candidates: list[dict[str, Any]]) -> str:
    for role in ("highres_tiff", "highres_jpeg", "highres", "primary"):
        for candidate in candidates:
            if candidate["downloadable"] and candidate["role"] == role:
                return sha1_text("|".join([media_key, candidate["role"], candidate["label"], candidate["url"]]))
    return ""


def _relationships(record: SmithsonianRecord) -> Iterable[RecordRelationshipProjection]:
    if record.linked_id:
        yield RecordRelationshipProjection(
            record_id=record.id,
            unit_code=record.unit_code,
            target=record.linked_id,
            relation_type="linked_id",
            label="",
            source="linkedId",
            position=0,
        )


def _iter_online_media(content: JsonObject) -> Iterable[Any]:
    descriptive = content.get("descriptiveNonRepeating")
    candidates: list[Any] = []
    if isinstance(descriptive, dict):
        candidates.extend(
            descriptive.get(key)
            for key in ("online_media", "onlineMedia", "online_media_group", "onlineMediaGroup", "media")
        )
    candidates.extend(content.get(key) for key in ("online_media", "onlineMedia", "media"))

    for candidate in candidates:
        if candidate is not None:
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


def _flatten_values(value: Any) -> Iterable[Any]:
    if isinstance(value, list):
        for item in value:
            yield from _flatten_values(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_values(item)
    else:
        yield value


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


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
        return value.strip()
    return str(value).strip()


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _identifier_type(label: str, category: str) -> str:
    normalized = _normalize_token(label) or _normalize_token(category)
    if "accession" in normalized:
        return "accession"
    if "catalog" in normalized:
        return "catalog"
    if "inventory" in normalized:
        return "inventory"
    if "number" in normalized:
        return "number"
    return normalized or "identifier"


def _parse_years(value: str) -> tuple[int | None, int | None, str]:
    text = value.strip()
    exact = re.fullmatch(r"(\d{4})", text)
    if exact:
        year = int(exact.group(1))
        return year, year, "year"
    decade = re.fullmatch(r"(\d{3})0s", text)
    if decade:
        start = int(f"{decade.group(1)}0")
        return start, start + 9, "decade"
    date_range = re.search(r"(\d{4})\D+(\d{4})", text)
    if date_range:
        return int(date_range.group(1)), int(date_range.group(2)), "range"
    return None, None, "text"


def _normalize_rights(value: str) -> str:
    normalized = _normalize_value(value)
    if normalized in {"cc0", "creative commons zero", "public domain"}:
        return "cc0"
    if "usage conditions apply" in normalized:
        return "usage_conditions_apply"
    if "no restrictions" in normalized:
        return "no_restrictions"
    return _normalize_token(value)


def _resource_role(label: str, url: str) -> str:
    lowered = f"{label} {url}".lower()
    if "thumbnail" in lowered or "thumb" in lowered:
        return "thumbnail"
    if "screen" in lowered:
        return "screen"
    if "tif" in lowered:
        return "highres_tiff"
    if "jp" in lowered or "jpeg" in lowered:
        return "highres_jpeg"
    if "high-resolution" in lowered:
        return "highres"
    return "resource"
