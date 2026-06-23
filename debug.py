from __future__ import annotations

import binascii
import hashlib
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TIFF_PATH = Path(r"C:\repos\smithsonian\NASM-A18890001000_PS01.tif")
KRITA_PNG_PATH = Path(r"C:\repos\smithsonian\NASM-A18890001000_PS01.png")
DATABASE_PATH = Path(r"C:\repos\smithsonian\data\smithsonian_scraper.sqlite3")

MAX_PIXELS = 8_388_608
JXL_QUALITY = 95
VERBOSE = os.getenv("SMITHSONIAN_DEBUG_VERBOSE", "").lower() in {"1", "true", "yes", "on"}


def main() -> int:
    print_header("Environment")
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"Working dir: {Path.cwd()}")
    print(f"NumPy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"cv2.haveImageReader(TIFF): {cv2.haveImageReader(str(TIFF_PATH))}")
    print(f"cv2.haveImageReader(Krita PNG): {cv2.haveImageReader(str(KRITA_PNG_PATH))}")
    print(f"cv2.haveImageWriter(.png): {cv2.haveImageWriter('.png')}")
    print_filtered_cv2_build_info()

    print_header("Input Files")
    inspect_file(TIFF_PATH)
    inspect_file(KRITA_PNG_PATH)

    print_header("TIFF Header")
    inspect_tiff_header(TIFF_PATH)

    print_header("Krita PNG")
    inspect_png_chunks(KRITA_PNG_PATH)

    print_header("OpenCV Read: TIFF")
    tiff_image = inspect_cv2_read(TIFF_PATH)

    print_header("OpenCV Read: Krita PNG")
    krita_image = inspect_cv2_read(KRITA_PNG_PATH)

    print_header("OpenCV Multi-Page TIFF")
    inspect_tiff_pages(TIFF_PATH)

    cjxl_path = find_tool("CJXL_PATH", "cjxl.exe", "cjxl")
    exiftool_path = find_tool("EXIFTOOL_PATH", "exiftool.exe", "exiftool")

    print_header("External Tools")
    print(f"cjxl: {cjxl_path or 'not found'}")
    print(f"exiftool: {exiftool_path or 'not found'}")

    print_header("Scraper Database")
    inspect_scraper_database(DATABASE_PATH)

    if exiftool_path and VERBOSE:
        print_header("ExifTool: TIFF")
        run_and_print([str(exiftool_path), "-G1", "-s", "-a", str(TIFF_PATH)], output_on_success=True)

        print_header("ExifTool: Krita PNG")
        run_and_print([str(exiftool_path), "-G1", "-s", "-a", str(KRITA_PNG_PATH)], output_on_success=True)

    debug_dir = Path(tempfile.mkdtemp(prefix="smithsonian-jxl-debug-"))
    print_header("Debug Output Directory")
    print(debug_dir)

    if tiff_image is None:
        print("OpenCV could not read the TIFF, so no OpenCV variants can be generated.")
    else:
        if cjxl_path:
            print_header("Scraper-Equivalent Path Test")
            test_scraper_equivalent_path(cjxl_path)

        generated_pngs = write_debug_png_variants(tiff_image, debug_dir)

        print_header("Generated PNG Inspection")
        for path in generated_pngs:
            inspect_file(path)
            inspect_png_chunks(path)
            inspect_cv2_read(path)

        if exiftool_path:
            print_header("Copy Metadata Onto OpenCV PNG")
            source = generated_pngs[0]
            copied = debug_dir / "opencv-unchanged-metadata-copied.png"
            shutil.copy2(source, copied)
            run_and_print(
                [
                    str(exiftool_path),
                    "-TagsFromFile",
                    str(TIFF_PATH),
                    "-EXIF:all",
                    "-IPTC:all",
                    "-XMP:all",
                    "-ICC_Profile",
                    "-overwrite_original",
                    str(copied),
                ]
            )
            inspect_file(copied)
            inspect_png_chunks(copied)
            generated_pngs.append(copied)

        if cjxl_path:
            print_header("cjxl Tests")
            candidates = [KRITA_PNG_PATH, *generated_pngs]
            for candidate in candidates:
                test_cjxl(cjxl_path, candidate, debug_dir)

    print_header("Read This First")
    print("The script now reports by exception: clean chunk CRCs and successful commands stay compact.")
    print("Set SMITHSONIAN_DEBUG_VERBOSE=1 to print full ExifTool and successful command output.")
    print("If generated OpenCV PNGs pass but the scraper-equivalent path fails, inspect path length, permissions, existing temp files, and security software around the media directory.")

    return 0


def print_header(title: str) -> None:
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def inspect_file(path: Path) -> None:
    print()
    print(f"Path: {path}")
    print(f"Exists: {path.exists()}")
    if not path.exists():
        return
    stat = path.stat()
    print(f"Size: {stat.st_size:,} bytes")
    print(f"SHA256: {sha256(path)}")
    with path.open("rb") as file:
        prefix = file.read(32)
    print(f"First 32 bytes: {prefix.hex(' ')}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_tiff_header(path: Path) -> None:
    if not path.exists():
        print("Missing TIFF")
        return

    with path.open("rb") as file:
        header = file.read(16)

    print(f"Raw: {header.hex(' ')}")
    if len(header) < 8:
        print("Too short for TIFF header")
        return

    endian_marker = header[:2]
    if endian_marker == b"II":
        endian = "<"
        print("Endian: little")
    elif endian_marker == b"MM":
        endian = ">"
        print("Endian: big")
    else:
        print("Not a classic TIFF header")
        return

    magic = struct.unpack(endian + "H", header[2:4])[0]
    print(f"Magic: {magic}")
    if magic == 42:
        first_ifd = struct.unpack(endian + "I", header[4:8])[0]
        print(f"Classic TIFF first IFD offset: {first_ifd}")
    elif magic == 43:
        print("BigTIFF detected")
    else:
        print("Unknown TIFF magic")


def inspect_cv2_read(path: Path) -> np.ndarray | None:
    print()
    print(f"OpenCV read: {path}")
    if not path.exists():
        print("Missing file")
        return None

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        print("cv2.imread returned None")
        return None

    print_array_info(image)

    if image.ndim == 2:
        channels = 1
    else:
        channels = image.shape[2]

    print(f"Interpreted channels: {channels}")
    print(f"C contiguous: {image.flags.c_contiguous}")
    print(f"Min: {safe_min(image)}")
    print(f"Max: {safe_max(image)}")
    print(f"Mean: {float(np.mean(image)):.4f}")

    if image.dtype == np.uint16:
        low = int(np.min(image))
        high = int(np.max(image))
        print(f"uint16 dynamic range: {low}..{high}")
        print(f"Would simple >>8 lose data? {'probably OK-ish' if high <= 65535 else 'unexpected'}")

    return image


def print_array_info(image: np.ndarray) -> None:
    print(f"shape: {image.shape}")
    print(f"dtype: {image.dtype}")
    print(f"ndim: {image.ndim}")
    print(f"size: {image.size:,}")


def safe_min(image: np.ndarray) -> Any:
    return image.min().item() if image.size else None


def safe_max(image: np.ndarray) -> Any:
    return image.max().item() if image.size else None


def inspect_tiff_pages(path: Path) -> None:
    try:
        ok, pages = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
    except Exception as exc:
        print(f"cv2.imreadmulti raised: {type(exc).__name__}: {exc}")
        return

    print(f"imreadmulti ok: {ok}")
    print(f"page count: {len(pages)}")
    for index, page in enumerate(pages[:10]):
        print(f"page {index}: shape={page.shape}, dtype={page.dtype}, min={safe_min(page)}, max={safe_max(page)}")
    if len(pages) > 10:
        print(f"... {len(pages) - 10} more pages")


def resize_dimensions(width: int, height: int, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    if width * height <= max_pixels:
        return width, height
    scale = (max_pixels / (width * height)) ** 0.5
    target_width = max(1, int(width * scale))
    target_height = max(1, int(height * scale))
    while target_width * target_height > max_pixels:
        if target_width >= target_height:
            target_width -= 1
        else:
            target_height -= 1
    return target_width, target_height


def write_debug_png_variants(image: np.ndarray, debug_dir: Path) -> list[Path]:
    print_header("Writing OpenCV Debug PNG Variants")

    height, width = image.shape[:2]
    target_width, target_height = resize_dimensions(width, height)
    print(f"Original dimensions: {width}x{height}")
    print(f"Target dimensions: {target_width}x{target_height}")

    resized = image
    if (target_width, target_height) != (width, height):
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        print("Resized with INTER_LANCZOS4")
        print_array_info(resized)

    variants: list[tuple[str, np.ndarray, list[int] | None]] = []

    variants.append(("opencv-unchanged.png", resized, None))

    plain_8bit = to_uint8(resized)
    variants.append(("opencv-plain-8bit-samechannels.png", plain_8bit, None))

    plain_3ch = force_3_channel_bgr(plain_8bit)
    variants.append(("opencv-plain-8bit-3ch.png", plain_3ch, None))

    gray_8bit = force_gray_8bit(plain_8bit)
    variants.append(("opencv-gray-8bit.png", gray_8bit, None))

    low_compression = [cv2.IMWRITE_PNG_COMPRESSION, 0]
    variants.append(("opencv-plain-8bit-3ch-compression0.png", plain_3ch, low_compression))

    written: list[Path] = []
    for filename, variant, params in variants:
        path = debug_dir / filename
        print()
        print(f"Writing: {path}")
        print_array_info(variant)
        if params is None:
            ok = cv2.imwrite(str(path), variant)
        else:
            ok = cv2.imwrite(str(path), variant, params)
        print(f"cv2.imwrite ok: {ok}")
        if ok:
            written.append(path)

    return written


def test_scraper_equivalent_path(cjxl_path: Path) -> None:
    resized_path = TIFF_PATH.with_name(f"{TIFF_PATH.stem}.resized.tmp.png")
    part_path = TIFF_PATH.with_name(f"{TIFF_PATH.stem}.tmp.jxl")
    output_path = TIFF_PATH.with_suffix(".jxl")

    print(f"Resized temp path: {resized_path}")
    print(f"Part JXL path:     {part_path}")
    print(f"Final JXL path:    {output_path}")
    for path in (resized_path, part_path, output_path):
        print(f"Pre-existing {path.name}: exists={path.exists()} size={path.stat().st_size if path.exists() else 0}")

    for path in (resized_path, part_path):
        try:
            if path.exists():
                path.unlink()
                print(f"Removed stale temp: {path}")
        except Exception as exc:
            print(f"TEMP CLEANUP EXCEPTION: {path}: {type(exc).__name__}: {exc}")

    try:
        write_resized_png_like_scraper(TIFF_PATH, resized_path)
    except Exception as exc:
        print(f"RESULT: FAIL before cjxl: {type(exc).__name__}: {exc}")
        return

    inspect_file(resized_path)
    inspect_png_chunks(resized_path)

    command = [str(cjxl_path), tool_path(resized_path), tool_path(part_path), "-q", str(JXL_QUALITY)]
    print(f"Command length: {len(subprocess.list2cmdline(command))} chars")
    print(f"Input path length: {len(str(resized_path))} chars")
    print(f"Output path length: {len(str(part_path))} chars")
    code, stdout, stderr = run_and_print(command, timeout=300.0, show_command=True, cwd=part_path.parent)
    produced = part_path.exists() and part_path.stat().st_size > 0
    print(f"Part output exists/non-empty: {produced}")
    if produced:
        print(f"Part output size: {part_path.stat().st_size:,} bytes")
        print(f"Part output SHA256: {sha256(part_path)}")

    if code == 0 and produced:
        print("RESULT: PASS scraper-equivalent path")
        print_cjxl_summary(stderr or stdout)
    else:
        print("RESULT: FAIL scraper-equivalent path")
        if stdout.strip():
            print("cjxl stdout:")
            print(stdout[:12000])
        if stderr.strip():
            print("cjxl stderr:")
            print(stderr[:12000])

    if os.getenv("SMITHSONIAN_DEBUG_KEEP_SCRAPER_TEMPS", "").lower() not in {"1", "true", "yes", "on"}:
        for path in (resized_path, part_path):
            try:
                if path.exists():
                    path.unlink()
                    print(f"Removed scraper-equivalent temp: {path}")
            except Exception as exc:
                print(f"TEMP CLEANUP EXCEPTION: {path}: {type(exc).__name__}: {exc}")


def write_resized_png_like_scraper(source_path: Path, resized_path: Path) -> None:
    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV could not read TIFF: {source_path}")
    height, width = image.shape[:2]
    target_width, target_height = resize_dimensions(width, height)
    if (target_width, target_height) != (width, height):
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    print(f"Writing scraper temp PNG beside TIFF: shape={image.shape}, dtype={image.dtype}")
    resized_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(resized_path), image)
    print(f"cv2.imwrite ok: {ok}")
    if not ok:
        raise RuntimeError(f"OpenCV could not write resized PNG: {resized_path}")
    if not resized_path.exists():
        raise RuntimeError(f"OpenCV reported success but file does not exist: {resized_path}")


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    if image.dtype == np.uint16:
        max_value = int(np.max(image)) if image.size else 0
        if max_value <= 255:
            return np.ascontiguousarray(image.astype(np.uint8))
        return np.ascontiguousarray((image / 257).clip(0, 255).astype(np.uint8))

    image_float = image.astype(np.float32)
    low = float(np.min(image_float)) if image_float.size else 0.0
    high = float(np.max(image_float)) if image_float.size else 1.0
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    normalized = (image_float - low) * (255.0 / (high - low))
    return np.ascontiguousarray(normalized.clip(0, 255).astype(np.uint8))


def force_3_channel_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    channels = image.shape[2]
    if channels == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if channels == 2:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return np.ascontiguousarray(image)
    if channels >= 4:
        return np.ascontiguousarray(image[:, :, :3])

    raise ValueError(f"Unsupported channel count: {channels}")


def force_gray_8bit(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.ascontiguousarray(image)
    channels = image.shape[2]
    if channels == 1:
        return np.ascontiguousarray(image[:, :, 0])
    if channels == 2:
        return np.ascontiguousarray(image[:, :, 0])
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if channels >= 4:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported channel count: {channels}")


def inspect_png_chunks(path: Path) -> None:
    print()
    print(f"PNG chunks: {path}")
    if not path.exists():
        print("Missing file")
        return

    with path.open("rb") as file:
        signature = file.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            print(f"Not a PNG signature: {signature.hex(' ')}")
            return

        index = 0
        idat_count = 0
        idat_bytes = 0
        chunk_counts: dict[str, int] = {}
        notable_chunks: list[str] = []
        exceptions: list[str] = []
        while True:
            raw_length = file.read(4)
            if len(raw_length) == 0:
                print("Reached EOF before IEND")
                return
            if len(raw_length) != 4:
                print("Truncated chunk length")
                return

            length = struct.unpack(">I", raw_length)[0]
            chunk_type = file.read(4)
            data = file.read(length)
            raw_crc = file.read(4)

            if len(chunk_type) != 4 or len(data) != length or len(raw_crc) != 4:
                print("Truncated chunk data")
                return

            expected_crc = struct.unpack(">I", raw_crc)[0]
            actual_crc = binascii.crc32(chunk_type)
            actual_crc = binascii.crc32(data, actual_crc) & 0xFFFFFFFF
            crc_ok = expected_crc == actual_crc
            name = chunk_type.decode("ascii", errors="replace")
            chunk_counts[name] = chunk_counts.get(name, 0) + 1

            extra = ""
            if name == "IHDR":
                extra = describe_ihdr(data)
            elif name in {"iCCP", "zTXt", "tEXt", "iTXt", "eXIf", "gAMA", "sRGB", "cHRM", "pHYs", "PLTE", "tRNS"}:
                extra = " notable"

            if name == "IDAT":
                idat_count += 1
                idat_bytes += length

            if not crc_ok:
                exceptions.append(f"{index:03d} {name} length={length:,} crc_ok=False")

            if name == "IHDR":
                print(f"IHDR:{extra}")
            elif name != "IDAT" and (extra or VERBOSE):
                notable_chunks.append(f"{index:03d} {name} length={length:,} crc_ok={crc_ok}{extra}")
            elif VERBOSE:
                print(f"{index:03d} {name} length={length:,} crc_ok={crc_ok}{extra}")

            index += 1
            if name == "IEND":
                print(f"Chunks: total={index}, IDAT={idat_count} chunks/{idat_bytes:,} bytes")
                print("Chunk types: " + ", ".join(f"{name}={count}" for name, count in sorted(chunk_counts.items())))
                if notable_chunks:
                    print("Notable non-IDAT chunks:")
                    for chunk in notable_chunks[:40]:
                        print(f"  {chunk}")
                    if len(notable_chunks) > 40:
                        print(f"  ... {len(notable_chunks) - 40} more hidden; set SMITHSONIAN_DEBUG_VERBOSE=1 to show all")
                if exceptions:
                    print("PNG EXCEPTIONS:")
                    for exception in exceptions:
                        print(f"  {exception}")
                else:
                    print("PNG exceptions: none")
                return


def describe_ihdr(data: bytes) -> str:
    if len(data) != 13:
        return " invalid IHDR length"

    width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(">IIBBBBB", data)
    color_types = {
        0: "grayscale",
        2: "truecolor RGB",
        3: "indexed color",
        4: "grayscale+alpha",
        6: "truecolor RGB+alpha",
    }
    return (
        f" width={width} height={height}"
        f" bit_depth={bit_depth}"
        f" color_type={color_type}({color_types.get(color_type, 'unknown')})"
        f" compression={compression}"
        f" filter={filter_method}"
        f" interlace={interlace}"
    )


def find_tool(env_name: str, *names: str) -> Path | None:
    configured = os.environ.get(env_name)
    if configured:
        path = Path(configured)
        if path.exists():
            return path
        found = shutil.which(configured)
        if found:
            return Path(found)

    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)

    return None


def tool_path(path: Path) -> str:
    return path.as_posix() if os.name == "nt" else str(path)


def inspect_scraper_database(path: Path) -> None:
    print(f"Path: {path}")
    print(f"Exists: {path.exists()}")
    if not path.exists():
        return

    try:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
    except Exception as exc:
        print(f"DB OPEN EXCEPTION: {type(exc).__name__}: {exc}")
        return

    try:
        print_db_counts(connection)
        print_recent_conversion_rows(connection)
        print_recent_failures(connection)
        print_matching_source_rows(connection, TIFF_PATH)
    finally:
        connection.close()


def print_db_counts(connection: sqlite3.Connection) -> None:
    for table in ("media_assets", "media_conversions", "failures"):
        try:
            rows = connection.execute(f"SELECT status, COUNT(*) AS count FROM {table} GROUP BY status").fetchall()
        except sqlite3.OperationalError:
            rows = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchall()
            print(f"{table}: total={rows[0]['count']}")
            continue
        if rows:
            print(f"{table}: " + ", ".join(f"{row['status']}={row['count']}" for row in rows))
        else:
            print(f"{table}: empty")


def print_recent_conversion_rows(connection: sqlite3.Connection) -> None:
    print()
    print("Recent non-complete conversions:")
    rows = connection.execute(
        """
        SELECT media_key, target_format, source_path, output_path, status, attempts, error, updated_at
        FROM media_conversions
        WHERE status != 'complete'
        ORDER BY updated_at DESC
        LIMIT 10
        """
    ).fetchall()
    if not rows:
        print("  none")
        return
    for row in rows:
        source_path = Path(str(row["source_path"]))
        print(f"  {row['status']} attempts={row['attempts']} key={row['media_key']} target={row['target_format']}")
        print(f"    source={source_path}")
        print(f"    source_exists={source_path.exists()} source_size={source_path.stat().st_size if source_path.exists() else 0} path_len={len(str(source_path))}")
        print(f"    output={row['output_path'] or '(empty)'}")
        print(f"    error={row['error'] or '(empty)'}")


def print_recent_failures(connection: sqlite3.Connection) -> None:
    print()
    print("Recent jxl-conversion failures:")
    rows = connection.execute(
        """
        SELECT identifier, payload_json, error, created_at
        FROM failures
        WHERE source = 'jxl-conversion'
        ORDER BY created_at DESC, id DESC
        LIMIT 10
        """
    ).fetchall()
    if not rows:
        print("  none")
        return
    for row in rows:
        print(f"  id={row['identifier']}")
        print(f"    error={row['error']}")
        payload = str(row["payload_json"])
        if payload:
            print(f"    payload={payload[:1000]}")


def print_matching_source_rows(connection: sqlite3.Connection, source_path: Path) -> None:
    print()
    print(f"Rows matching source filename {source_path.name}:")
    rows = connection.execute(
        """
        SELECT media_key, target_format, source_path, output_path, status, attempts, error
        FROM media_conversions
        WHERE source_path LIKE ?
        ORDER BY updated_at DESC
        LIMIT 10
        """,
        (f"%{source_path.name}",),
    ).fetchall()
    if not rows:
        print("  none")
        return
    for row in rows:
        print(f"  {row['status']} attempts={row['attempts']} key={row['media_key']} target={row['target_format']}")
        print(f"    source={row['source_path']}")
        print(f"    output={row['output_path'] or '(empty)'}")
        print(f"    error={row['error'] or '(empty)'}")


def run_and_print(
    command: list[str],
    timeout: float = 120.0,
    *,
    output_on_success: bool = False,
    show_command: bool = True,
    cwd: Path | None = None,
) -> tuple[int, str, str]:
    print()
    if show_command:
        print("Command:")
        print(" ".join(f'"{part}"' if " " in part else part for part in command))
    if cwd is not None:
        print(f"cwd: {cwd}")

    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            timeout=timeout,
            cwd=cwd,
        )
    except Exception as exc:
        print(f"FAILED TO START/RUN: {type(exc).__name__}: {exc}")
        return 999, "", str(exc)

    print(f"Exit code: {completed.returncode}")
    should_print_output = output_on_success or completed.returncode != 0 or VERBOSE
    if should_print_output and completed.stdout.strip():
        print("STDOUT:")
        print(completed.stdout[:12000])
    if should_print_output and completed.stderr.strip():
        print("STDERR:")
        print(completed.stderr[:12000])
    if not should_print_output and (completed.stdout.strip() or completed.stderr.strip()):
        print("Output: hidden because command succeeded; set SMITHSONIAN_DEBUG_VERBOSE=1 to show it.")

    return completed.returncode, completed.stdout, completed.stderr


def test_cjxl(cjxl_path: Path, input_png: Path, debug_dir: Path) -> None:
    output_jxl = debug_dir / f"{input_png.stem}.jxl"

    print()
    print("-" * 88)
    print(f"Testing cjxl input: {input_png}")
    inspect_png_summary_only(input_png)

    command = [
        str(cjxl_path),
        "-v",
        "-v",
        str(input_png),
        str(output_jxl),
        "-q",
        str(JXL_QUALITY),
    ]
    code, stdout, stderr = run_and_print(command, timeout=300.0)

    print(f"Output exists: {output_jxl.exists()}")
    if output_jxl.exists():
        print(f"Output size: {output_jxl.stat().st_size:,} bytes")
        print(f"Output SHA256: {sha256(output_jxl)}")

    accepted = code == 0 and output_jxl.exists() and output_jxl.stat().st_size > 0
    if accepted:
        print("RESULT: PASS")
        print_cjxl_summary(stderr or stdout)
    else:
        print("RESULT: FAIL")
        if stdout.strip():
            print("cjxl stdout:")
            print(stdout[:12000])
        if stderr.strip():
            print("cjxl stderr:")
            print(stderr[:12000])


def print_cjxl_summary(output: str) -> None:
    interesting_prefixes = ("JPEG XL encoder", "Read ", "Compressed to")
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(interesting_prefixes):
            print(f"cjxl: {stripped}")


def inspect_png_summary_only(path: Path) -> None:
    if not path.exists():
        print("Missing PNG")
        return

    with path.open("rb") as file:
        signature = file.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            print("Not PNG")
            return
        raw_length = file.read(4)
        chunk_type = file.read(4)
        length = struct.unpack(">I", raw_length)[0]
        data = file.read(length)
        if chunk_type == b"IHDR":
            print(f"IHDR:{describe_ihdr(data)}")


def print_filtered_cv2_build_info() -> None:
    print()
    print("OpenCV build info lines mentioning image codecs:")
    try:
        build_info = cv2.getBuildInformation()
    except Exception as exc:
        print(f"Could not get build info: {exc}")
        return

    interesting_terms = [
        "PNG",
        "TIFF",
        "JPEG",
        "JPEG 2000",
        "WEBP",
        "OpenEXR",
        "ZLIB",
        "Image I/O",
        "Media I/O",
    ]

    for line in build_info.splitlines():
        upper = line.upper()
        if any(term.upper() in upper for term in interesting_terms):
            print(line)


if __name__ == "__main__":
    raise SystemExit(main())