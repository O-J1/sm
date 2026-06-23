from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from .models import MediaDownload, SmithsonianRecord, sha1_text, stable_file_stem


class RecordStore:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def record_path(self, record: SmithsonianRecord) -> Path:
        unit = stable_file_stem(record.unit_code or "UNKNOWN")
        return self.output_dir / "metadata" / unit / "records.jsonl"

    def append_record(self, record: SmithsonianRecord) -> Path:
        path = self.record_path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.raw, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        return path

    def media_path(
        self,
        media: MediaDownload,
        content_type: str = "",
        content_disposition: str = "",
        first_bytes: bytes = b"",
    ) -> Path:
        unit = stable_file_stem(media.unit_code or "UNKNOWN")
        hash_prefix = (media.record_hash or media.key)[:2]
        extension = (
            _extension_from_content_disposition(content_disposition)
            or _extension_from_content_type(content_type)
            or _extension_from_magic(first_bytes)
            or _extension_from_url(media.url)
            or ".bin"
        )
        source = media.url or media.guid
        label = stable_file_stem(media.resource_label or media.kind)
        stem = stable_file_stem(f"{media.record_id}_{media.kind}_{label}_{sha1_text(source)[:12]}")
        return self.output_dir / "media" / unit / hash_prefix / f"{stem}{extension}"

    def media_part_path(self, media: MediaDownload) -> Path:
        unit = stable_file_stem(media.unit_code or "UNKNOWN")
        hash_prefix = (media.record_hash or media.key)[:2]
        source = media.url or media.guid
        label = stable_file_stem(media.resource_label or media.kind)
        stem = stable_file_stem(f"{media.record_id}_{media.kind}_{label}_{sha1_text(source)[:12]}")
        return self.output_dir / "media" / unit / hash_prefix / f"{stem}.part"

    def conversion_path(self, source_path: Path, suffix: str = ".jxl") -> Path:
        normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return source_path.with_suffix(normalized_suffix)

    def conversion_part_path(self, source_path: Path, suffix: str = ".jxl") -> Path:
        normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return source_path.with_name(f"{source_path.stem}.tmp{normalized_suffix}")


def _extension_from_url(url: str) -> str:
    parsed = urlparse(url)
    suffix = Path(unquote(parsed.path)).suffix.lower()
    if suffix and len(suffix) <= 10:
        return suffix
    return ""


def _extension_from_content_disposition(content_disposition: str) -> str:
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)', content_disposition, re.IGNORECASE)
    if not match:
        return ""
    suffix = Path(unquote(match.group(1))).suffix.lower()
    if suffix and len(suffix) <= 10:
        return suffix
    return ""


def _extension_from_content_type(content_type: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    return {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/tiff": ".tif",
        "application/pdf": ".pdf",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "video/mp4": ".mp4",
        "text/plain": ".txt",
    }.get(media_type, "")


def _extension_from_magic(first_bytes: bytes) -> str:
    if first_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if first_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if first_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if first_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return ".tif"
    if first_bytes.startswith(b"%PDF"):
        return ".pdf"
    return ""