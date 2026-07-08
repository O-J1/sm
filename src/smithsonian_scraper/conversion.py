from __future__ import annotations

import asyncio
import json
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable

try:
    import cv2
except Exception as cv2_import_error:
    _CV2_IMPORT_ERROR = cv2_import_error

    class _MissingCv2:
        IMREAD_UNCHANGED = -1
        INTER_LANCZOS4 = 4

        def imread(self, *args, **kwargs):
            raise RuntimeError(_missing_cv2_message()) from _CV2_IMPORT_ERROR

        def resize(self, *args, **kwargs):
            raise RuntimeError(_missing_cv2_message()) from _CV2_IMPORT_ERROR

        def imwrite(self, *args, **kwargs):
            raise RuntimeError(_missing_cv2_message()) from _CV2_IMPORT_ERROR

    cv2 = _MissingCv2()


def _missing_cv2_message() -> str:
    return (
        "opencv-python-headless<4.9.0 is required for conversion and must be installed in a compatible "
        "Python environment. Use Python 3.11 or 3.12 for this project; Python 3.13 is not supported by "
        "the pinned OpenCV wheel."
    )

from .storage import RecordStore


MAX_CONVERSION_PIXELS = 8_388_608
DEFAULT_JXL_QUALITY = 95
DEFAULT_DERIVATIVE_QUALITY = 90
MAX_COMMAND_ERROR_OUTPUT = 8000
METADATA_GROUP_ARGS = (
    "-EXIF:all",
    "-IPTC:all",
    "-XMP:all",
    "-ICC_Profile",
)
IGNORED_METADATA_TAGS = {
    "bitspersample",
    "colorspace",
    "compression",
    "exifimageheight",
    "exifimagewidth",
    "imageheight",
    "imagelength",
    "imagesize",
    "imagewidth",
    "megapixels",
    "mimetype",
    "photometricinterpretation",
    "pixelydimension",
    "pixelxdimension",
    "samplesperpixel",
    "sourcefile",
    "ycbcrsubsampling",
}


class JxlConverter:
    def __init__(
        self,
        *,
        cjxl_path: Path,
        exiftool_path: Path,
        store: RecordStore,
        max_pixels: int = MAX_CONVERSION_PIXELS,
        quality: int = DEFAULT_JXL_QUALITY,
        cjxl_verbose: int = 0,
        skip_metadata: bool = False,
        timeout: float | None = None,
        progress: Callable[[str], None] | None = None,
        magick_path: Path | None = None,
    ) -> None:
        self._cjxl_path = cjxl_path
        self._exiftool_path = exiftool_path
        self._store = store
        self._max_pixels = max_pixels
        self._quality = quality
        self._cjxl_verbose = cjxl_verbose
        self._skip_metadata = skip_metadata
        self._timeout = timeout
        self._progress = progress
        self._magick_path = magick_path

    def metadata_copy_command(self, source_path: Path, target_path: Path) -> list[str]:
        return [
            str(self._exiftool_path),
            "-TagsFromFile",
            _tool_path(source_path),
            *METADATA_GROUP_ARGS,
            "-overwrite_original",
            _tool_path(target_path),
        ]

    def metadata_json_command(self, path: Path) -> list[str]:
        return [str(self._exiftool_path), "-j", "-G1", "-s", *METADATA_GROUP_ARGS, _tool_path(path)]

    def strip_icc_command(self, path: Path) -> list[str]:
        return [str(self._exiftool_path), "-ICC_Profile:all=", "-overwrite_original", _tool_path(path)]

    def encode_command(self, source_path: Path, output_path: Path, quality: int | None = None) -> list[str]:
        verbose_args = ["-v"] * self._cjxl_verbose
        effective_quality = self._quality if quality is None else quality
        return [str(self._cjxl_path), *verbose_args, _tool_path(source_path), _tool_path(output_path), "-q", str(effective_quality)]

    def derivative_command(self, source_path: Path, output_path: Path, edge: int, quality: int) -> list[str]:
        """ImageMagick Lanczos2Sharp resize to `edge` on the longest side
        (aspect preserved, never upscaled) generated from the original file."""
        if self._magick_path is None:
            raise RuntimeError(
                "derivative outputs require ImageMagick: set MAGICK_PATH or pass --magick-path "
                "(user-space install, e.g. the extracted GitHub-release AppImage)"
            )
        return [
            str(self._magick_path),
            _tool_path(source_path),
            "-filter",
            "Lanczos2Sharp",
            "-resize",
            f"{edge}x{edge}>",
            "-quality",
            str(quality),
            _tool_path(output_path),
        ]

    async def convert_row(self, row: sqlite3.Row) -> tuple[Path, int]:
        source_path = Path(str(row["source_path"]))
        output_kind = str(_row_value(row, "output_kind") or "highres")
        planned_output = str(_row_value(row, "output_path") or "")
        if output_kind != "highres":
            return await self._convert_derivative(row, source_path, planned_output, output_kind)
        if planned_output:
            output_path = self._store.output_dir / Path(planned_output)
        else:
            output_path = self._store.conversion_path(source_path, f".{row['target_format']}")
        if output_path.exists() and output_path.stat().st_size > 0:
            self._log(f"conversion skipped existing output: {output_path} ({output_path.stat().st_size} bytes)")
            return output_path, output_path.stat().st_size

        if not source_path.exists():
            raise FileNotFoundError(f"source TIFF does not exist: {source_path}")

        resized_path = _resized_temp_path(source_path)
        part_path = self._store.conversion_part_path(source_path, f".{row['target_format']}")
        _remove_paths(resized_path, part_path)

        try:
            started_at = time.monotonic()
            self._log(f"conversion resize start: {source_path}")
            max_pixels = int(_row_value(row, "target_max_pixels") or self._max_pixels)
            quality = _row_value(row, "quality")
            self._write_resized_png(source_path, resized_path, max_pixels)
            self._log(f"conversion resize complete: {resized_path} ({resized_path.stat().st_size} bytes)")
            if self._skip_metadata:
                self._log(f"conversion metadata copy skipped: {resized_path}")
            else:
                self._log(f"conversion metadata copy start: {resized_path}")
                try:
                    await _run_command(self.metadata_copy_command(source_path, resized_path), timeout=self._timeout)
                except Exception as exc:
                    raise RuntimeError(f"metadata copy failed for {resized_path}: {exc}") from exc
                self._log(f"conversion metadata copy complete: {resized_path}")
            self._log(f"conversion cjxl start: {resized_path} -> {part_path}")
            icc_stripped = False
            try:
                await _run_command(
                    self.encode_command(resized_path.resolve(), part_path.resolve(), quality),
                    timeout=self._timeout,
                    cwd=part_path.parent.resolve(),
                )
            except Exception as exc:
                if "JxlEncoderSetICCProfile" not in str(exc):
                    raise RuntimeError(f"cjxl failed for {resized_path}: {exc}") from exc
                # Corrupt or unsupported ICC profile. Strip it and retry once:
                # pixels are unaffected, the image falls back to sRGB
                # interpretation (which is all a broken profile provided anyway).
                self._log(f"conversion cjxl ICC profile rejected; stripping and retrying: {resized_path}")
                try:
                    await _run_command(self.strip_icc_command(resized_path), timeout=self._timeout)
                    await _run_command(
                        self.encode_command(resized_path.resolve(), part_path.resolve(), quality),
                        timeout=self._timeout,
                        cwd=part_path.parent.resolve(),
                    )
                except Exception as retry_exc:
                    raise RuntimeError(
                        f"cjxl failed for {resized_path} even after ICC strip: {retry_exc}"
                    ) from retry_exc
                icc_stripped = True
                self._log(f"conversion recovered after ICC strip: {resized_path}")
            if not part_path.exists() or part_path.stat().st_size == 0:
                raise RuntimeError("cjxl completed without producing a non-empty JXL")
            self._log(f"conversion cjxl complete: {part_path} ({part_path.stat().st_size} bytes)")
            if self._skip_metadata:
                self._log(f"conversion metadata verify skipped: {part_path}")
            else:
                self._log(f"conversion metadata verify start: {part_path}")
                await self._verify_metadata(source_path, part_path, ignore_icc=icc_stripped)
                self._log(f"conversion metadata verify complete: {part_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(part_path, output_path)
            elapsed = time.monotonic() - started_at
            self._log(f"conversion complete: {output_path} ({output_path.stat().st_size} bytes, {elapsed:.1f}s)")
            return output_path, output_path.stat().st_size
        finally:
            _remove_paths(resized_path, part_path)

    async def _convert_derivative(
        self,
        row: sqlite3.Row,
        source_path: Path,
        planned_output: str,
        output_kind: str,
    ) -> tuple[Path, int]:
        """Produce a small Lanczos2Sharp pretraining resize from the ORIGINAL
        source file (never from the downscaled highres output)."""
        edge = int(_row_value(row, "target_edge") or 0)
        if edge < 1:
            raise RuntimeError(f"derivative row without target_edge: {row['resource_key']} ({output_kind})")
        if planned_output:
            output_path = self._store.output_dir / Path(planned_output)
        else:
            extension = str(_row_value(row, "output_format") or "jpg")
            output_path = self._store.conversion_path(
                source_path.with_name(f"{source_path.stem}_{edge}{source_path.suffix}"), f".{extension}"
            )
        if output_path.exists() and output_path.stat().st_size > 0:
            self._log(f"derivative skipped existing output: {output_path} ({output_path.stat().st_size} bytes)")
            return output_path, output_path.stat().st_size
        if not source_path.exists():
            raise FileNotFoundError(f"source image does not exist: {source_path}")
        quality = int(_row_value(row, "quality") or DEFAULT_DERIVATIVE_QUALITY)
        part_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _remove_paths(part_path)
        try:
            started_at = time.monotonic()
            self._log(f"derivative start: {source_path} -> {output_path} ({edge}px)")
            await _run_command(
                self.derivative_command(source_path.resolve(), part_path.resolve(), edge, quality),
                timeout=self._timeout,
                cwd=part_path.parent.resolve(),
            )
            if not part_path.exists() or part_path.stat().st_size == 0:
                raise RuntimeError("magick completed without producing a non-empty output")
            os.replace(part_path, output_path)
            elapsed = time.monotonic() - started_at
            self._log(f"derivative complete: {output_path} ({output_path.stat().st_size} bytes, {elapsed:.1f}s)")
            return output_path, output_path.stat().st_size
        finally:
            _remove_paths(part_path)

    def _log(self, message: str) -> None:
        if self._progress is not None:
            self._progress(message)

    def _write_resized_png(self, source_path: Path, resized_path: Path, max_pixels: int | None = None) -> None:
        image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"OpenCV could not read TIFF: {source_path}")
        height, width = image.shape[:2]
        target_width, target_height = resize_dimensions(width, height, max_pixels or self._max_pixels)
        if (target_width, target_height) != (width, height):
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        resized_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(resized_path), image):
            raise RuntimeError(f"OpenCV could not write resized PNG: {resized_path}")

    async def _verify_metadata(self, source_path: Path, output_path: Path, *, ignore_icc: bool = False) -> None:
        source_metadata = await self._selected_metadata(source_path)
        output_metadata = await self._selected_metadata(output_path)
        if ignore_icc:
            source_metadata = {k: v for k, v in source_metadata.items() if not k.startswith("ICC_Profile:")}
            output_metadata = {k: v for k, v in output_metadata.items() if not k.startswith("ICC_Profile:")}
        if source_metadata != output_metadata:
            missing = sorted(set(source_metadata) - set(output_metadata))[:10]
            changed = sorted(key for key in source_metadata if key in output_metadata and source_metadata[key] != output_metadata[key])[:10]
            details = []
            if missing:
                details.append(f"missing={missing}")
            if changed:
                details.append(f"changed={changed}")
            raise RuntimeError("metadata verification failed" + (f": {', '.join(details)}" if details else ""))

    async def _selected_metadata(self, path: Path) -> dict[str, Any]:
        stdout, _ = await _run_command(self.metadata_json_command(path), timeout=self._timeout)
        try:
            payload = json.loads(stdout.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"ExifTool returned invalid JSON for {path}") from exc
        if not payload:
            return {}
        if not isinstance(payload[0], dict):
            raise RuntimeError(f"ExifTool returned unexpected JSON for {path}")
        return _normalize_metadata(payload[0])


def resize_dimensions(width: int, height: int, max_pixels: int = MAX_CONVERSION_PIXELS) -> tuple[int, int]:
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if max_pixels < 1:
        raise ValueError("max_pixels must be positive")
    if width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / (width * height))
    target_width = max(1, math.floor(width * scale))
    target_height = max(1, math.floor(height * scale))
    while target_width * target_height > max_pixels:
        if target_width >= target_height:
            target_width -= 1
        else:
            target_height -= 1
    return target_width, target_height


async def _run_command(command: list[str], *, timeout: float | None, cwd: Path | None = None) -> tuple[bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        if timeout is None:
            stdout, stderr = await process.communicate()
        else:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.communicate()
        raise TimeoutError(f"command timed out after {timeout} seconds: {command[0]}") from exc

    if process.returncode != 0:
        output = _command_output(stdout, stderr)
        if output:
            raise RuntimeError(f"{command[0]} failed with exit code {process.returncode}: {output[:MAX_COMMAND_ERROR_OUTPUT]}")
        raise RuntimeError(f"{command[0]} failed with exit code {process.returncode}")
    return stdout, stderr


def _command_output(stdout: bytes, stderr: bytes) -> str:
    parts = []
    if stdout:
        parts.append("stdout:\n" + stdout.decode("utf-8", errors="replace").strip())
    if stderr:
        parts.append("stderr:\n" + stderr.decode("utf-8", errors="replace").strip())
    return "\n".join(part for part in parts if part.strip())


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for raw_key, value in metadata.items():
        key = str(raw_key)
        tag = key.rsplit(":", 1)[-1]
        if tag.lower() in IGNORED_METADATA_TAGS:
            continue
        if not key.startswith(("EXIF:", "IPTC:", "XMP:", "ICC_Profile:")):
            continue
        normalized[key] = value
    return normalized


def _tool_path(path: Path) -> str:
    return path.as_posix() if os.name == "nt" else str(path)


def _row_value(row, key: str, default=None):
    """Tolerant column access for sqlite3.Row / mapping rows that may predate
    the policy columns."""
    try:
        keys = row.keys()
    except AttributeError:
        return default
    if key not in keys:
        return default
    return row[key]


def _resized_temp_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}.resized.tmp.png")


def _remove_paths(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            path.unlink()