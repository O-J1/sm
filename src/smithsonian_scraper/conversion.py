from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path

from .storage import RecordStore


JXL_DARKTABLE_ARGS = (
    "--out-ext",
    "jxl",
    "--hq",
    "true",
    "--apply-custom-presets",
    "false",
    "--core",
    "--conf",
    "plugins/imageio/format/jxl/bpp=8",
    "--conf",
    "plugins/imageio/format/jxl/quality=100",
    "--conf",
    "plugins/imageio/format/jxl/original=1",
    "--conf",
    "plugins/imageio/format/jxl/effort=9",
    "--conf",
    "plugins/imageio/format/jxl/tier=0",
)


class DarktableConverter:
    def __init__(self, *, darktable_cli_path: Path, store: RecordStore, timeout: float | None = None) -> None:
        self._darktable_cli_path = darktable_cli_path
        self._store = store
        self._timeout = timeout

    def command(self, source_path: Path, output_path: Path) -> list[str]:
        return [str(self._darktable_cli_path), _darktable_path(source_path), _darktable_path(output_path), *JXL_DARKTABLE_ARGS]

    async def convert_row(self, row: sqlite3.Row) -> tuple[Path, int]:
        source_path = Path(str(row["source_path"]))
        output_path = self._store.conversion_path(source_path, f".{row['target_format']}")
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path, output_path.stat().st_size

        if not source_path.exists():
            raise FileNotFoundError(f"source TIFF does not exist: {source_path}")

        part_path = self._store.conversion_part_path(source_path, f".{row['target_format']}")
        part_path.parent.mkdir(parents=True, exist_ok=True)
        _remove_temp_outputs(part_path)

        command = self.command(source_path.resolve(), Path(part_path.name))
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=part_path.parent,
        )
        try:
            if self._timeout is None:
                stdout, stderr = await process.communicate()
            else:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self._timeout)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            _remove_temp_outputs(part_path)
            raise TimeoutError(f"darktable-cli timed out after {self._timeout} seconds") from exc

        if process.returncode != 0:
            _remove_temp_outputs(part_path)
            raise RuntimeError(_process_error(process.returncode, stdout, stderr))

        produced_path = _completed_temp_output(part_path)
        if produced_path is None:
            _remove_temp_outputs(part_path)
            raise RuntimeError("darktable-cli completed without producing a non-empty JXL")

        os.replace(produced_path, output_path)
        _remove_temp_outputs(part_path)
        return output_path, output_path.stat().st_size


def _process_error(returncode: int | None, stdout: bytes, stderr: bytes) -> str:
    output = (stderr or stdout).decode("utf-8", errors="replace").strip()
    if output:
        return f"darktable-cli failed with exit code {returncode}: {output[:1000]}"
    return f"darktable-cli failed with exit code {returncode}"


def _darktable_path(path: Path) -> str:
    return path.as_posix() if os.name == "nt" else str(path)


def _temp_output_candidates(part_path: Path) -> list[Path]:
    return [part_path, *sorted(part_path.parent.glob(f"{part_path.stem}_*{part_path.suffix}"))]


def _completed_temp_output(part_path: Path) -> Path | None:
    for candidate in _temp_output_candidates(part_path):
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def _remove_temp_outputs(part_path: Path) -> None:
    for candidate in _temp_output_candidates(part_path):
        if candidate.exists():
            candidate.unlink()
