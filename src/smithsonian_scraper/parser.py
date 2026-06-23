from __future__ import annotations

from typing import Any

from .models import JsonObject, SmithsonianRecord


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