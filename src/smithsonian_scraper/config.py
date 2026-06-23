from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_BASE_URL = "https://api.si.edu/openaccess/api/v1.0"


@dataclass(frozen=True)
class ScraperConfig:
    api_key: str
    output_dir: Path = Path("data")
    database_path: Path = Path("data/smithsonian_scraper.sqlite3")
    base_url: str = DEFAULT_BASE_URL
    metadata_workers: int = 4
    media_workers: int = 2
    http_connections: int = 6
    rows_per_page: int = 1000
    record_limit: int | None = None
    retry_limit: int = 5
    request_timeout: float = 60.0
    conversion_workers: int = 1
    conversion_backlog_limit: int = 2
    conversion_timeout: float | None = None
    conversion_max_pixels: int = 8_388_608
    jxl_quality: int = 95
    cjxl_verbose: int = 0
    skip_conversion_metadata: bool = False
    cjxl_path: Path | None = None
    exiftool_path: Path | None = None
    query: str = "*:*"
    sort: str = "id"
    record_type: str = "edanmdm"
    row_group: str = "objects"
    include_units: tuple[str, ...] = field(default_factory=tuple)
    exclude_units: tuple[str, ...] = field(default_factory=tuple)
    download_media: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.rows_per_page <= 1000:
            raise ValueError("rows_per_page must be between 1 and 1000")
        if self.record_limit is not None and self.record_limit < 1:
            raise ValueError("record_limit must be at least 1 when provided")
        if self.metadata_workers < 1:
            raise ValueError("metadata_workers must be at least 1")
        if self.media_workers < 0:
            raise ValueError("media_workers cannot be negative")
        if self.http_connections < 1:
            raise ValueError("http_connections must be at least 1")
        if self.conversion_workers < 1:
            raise ValueError("conversion_workers must be at least 1")
        if self.conversion_backlog_limit < 1:
            raise ValueError("conversion_backlog_limit must be at least 1")
        if self.conversion_timeout is not None and self.conversion_timeout <= 0:
            raise ValueError("conversion_timeout must be greater than 0 when provided")
        if self.conversion_max_pixels < 1:
            raise ValueError("conversion_max_pixels must be at least 1")
        if not 1 <= self.jxl_quality <= 100:
            raise ValueError("jxl_quality must be between 1 and 100")
        if self.cjxl_verbose < 0:
            raise ValueError("cjxl_verbose cannot be negative")


def build_config(
    *,
    api_key: str | None = None,
    output_dir: str | Path = "data",
    database_path: str | Path | None = None,
    metadata_workers: int = 4,
    media_workers: int = 2,
    http_connections: int = 6,
    rows_per_page: int = 1000,
    record_limit: int | None = None,
    retry_limit: int = 5,
    request_timeout: float = 60.0,
    conversion_workers: int | None = None,
    conversion_backlog_limit: int | None = None,
    conversion_timeout: float | None = None,
    conversion_max_pixels: int | None = None,
    jxl_quality: int | None = None,
    cjxl_verbose: int | None = None,
    skip_conversion_metadata: bool = False,
    cjxl_path: str | Path | None = None,
    exiftool_path: str | Path | None = None,
    query: str = "*:*",
    sort: str = "id",
    record_type: str = "edanmdm",
    row_group: str = "objects",
    include_units: tuple[str, ...] = (),
    exclude_units: tuple[str, ...] = (),
    download_media: bool = True,
) -> ScraperConfig:
    output_path = Path(output_dir)
    db_path = Path(database_path) if database_path else output_path / "smithsonian_scraper.sqlite3"
    configured_cjxl = cjxl_path or os.getenv("CJXL_PATH") or None
    configured_exiftool = exiftool_path or os.getenv("EXIFTOOL_PATH") or None
    return ScraperConfig(
        api_key=api_key or os.getenv("SMITHSONIAN_API_KEY", ""),
        output_dir=output_path,
        database_path=db_path,
        metadata_workers=metadata_workers,
        media_workers=media_workers,
        http_connections=http_connections,
        rows_per_page=rows_per_page,
        record_limit=record_limit,
        retry_limit=retry_limit,
        request_timeout=request_timeout,
        conversion_workers=conversion_workers
        if conversion_workers is not None
        else _env_int("SMITHSONIAN_CONVERSION_WORKERS", 1),
        conversion_backlog_limit=conversion_backlog_limit
        if conversion_backlog_limit is not None
        else _env_int("SMITHSONIAN_CONVERSION_BACKLOG_LIMIT", 2),
        conversion_timeout=conversion_timeout
        if conversion_timeout is not None
        else _env_float("SMITHSONIAN_CONVERSION_TIMEOUT"),
        conversion_max_pixels=conversion_max_pixels
        if conversion_max_pixels is not None
        else _env_int("SMITHSONIAN_CONVERSION_MAX_PIXELS", 8_388_608),
        jxl_quality=jxl_quality if jxl_quality is not None else _env_int("SMITHSONIAN_JXL_QUALITY", 95),
        cjxl_verbose=cjxl_verbose if cjxl_verbose is not None else _env_int("SMITHSONIAN_CJXL_VERBOSE", 0),
        skip_conversion_metadata=skip_conversion_metadata,
        cjxl_path=Path(configured_cjxl) if configured_cjxl else None,
        exiftool_path=Path(configured_exiftool) if configured_exiftool else None,
        query=query,
        sort=sort,
        record_type=record_type,
        row_group=row_group,
        include_units=include_units,
        exclude_units=exclude_units,
        download_media=download_media,
    )


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return float(value)