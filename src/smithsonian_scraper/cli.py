from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from .api import SmithsonianClient
from .config import ScraperConfig, build_config
from .conversion import JxlConverter
from .media import MediaDownloader
from .scheduler import CrawlScheduler
from .state import StateStore
from .storage import RecordStore


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
        if args.command == "probe":
            asyncio.run(_probe(config))
        elif args.command in {"scrape", "resume"}:
            asyncio.run(_scrape(config, download_media_after=args.download_media))
        elif args.command == "download-media":
            asyncio.run(_download_media(config))
        elif args.command == "convert-media":
            asyncio.run(_convert_media(config))
        elif args.command == "status":
            asyncio.run(_status(config))
        else:
            parser.print_help()
            return 2
    except KeyboardInterrupt:
        print("Interrupted; checkpointed work can be resumed.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


async def _probe(config: ScraperConfig) -> None:
    _require_api_key(config)
    state = StateStore(config.database_path)
    await state.initialize()
    client = SmithsonianClient(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.request_timeout,
        retry_limit=config.retry_limit,
        max_connections=config.http_connections,
    )
    try:
        scheduler = CrawlScheduler(config=config, client=client, state=state, store=RecordStore(config.output_dir))
        result = await scheduler.probe()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        await client.close()
        await state.close()


async def _scrape(config: ScraperConfig, *, download_media_after: bool) -> None:
    _require_api_key(config)
    state = StateStore(config.database_path)
    await state.initialize()
    client = SmithsonianClient(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.request_timeout,
        retry_limit=config.retry_limit,
        max_connections=config.http_connections,
    )
    try:
        scheduler = CrawlScheduler(config=config, client=client, state=state, store=RecordStore(config.output_dir))
        await scheduler.scrape()
    finally:
        await client.close()

    if download_media_after and config.download_media:
        await _download_media(config, existing_state=state)
    print(json.dumps(await state.status_counts(), indent=2, sort_keys=True))
    await state.close()


async def _download_media(config: ScraperConfig, existing_state: StateStore | None = None) -> None:
    state = existing_state or StateStore(config.database_path)
    if existing_state is None:
        await state.initialize()
    conversion_wake = asyncio.Event()
    downloads_done = asyncio.Event()
    conversion_task = None
    if _conversion_enabled(config):
        conversion_task = asyncio.create_task(
            _run_conversion_worker(config, state, wake_event=conversion_wake, stop_event=downloads_done)
        )
    downloader = MediaDownloader(
        store=RecordStore(config.output_dir),
        timeout=config.request_timeout,
        retry_limit=config.retry_limit,
        max_connections=max(config.media_workers, 1),
    )
    download_error: BaseException | None = None
    close_error: BaseException | None = None
    conversion_error: BaseException | None = None
    try:
        try:
            pending_downloads: dict[asyncio.Task[tuple[Path, int]], object] = {}
            media_worker_count = max(config.media_workers, 1)
            downloads_exhausted = False
            while pending_downloads or not downloads_exhausted:
                can_schedule_downloads = not downloads_exhausted and len(pending_downloads) < media_worker_count
                if can_schedule_downloads and conversion_task is not None:
                    backlog_count = await state.conversion_backlog_count(max_attempts=config.retry_limit)
                    can_schedule_downloads = backlog_count < config.conversion_backlog_limit

                if can_schedule_downloads:
                    rows = await state.next_media_batch(
                        media_worker_count - len(pending_downloads),
                        max_attempts=config.retry_limit,
                    )
                    downloads_exhausted = not rows
                    for row in rows:
                        pending_downloads[asyncio.create_task(downloader.download_row(row))] = row

                if not pending_downloads:
                    if not downloads_exhausted:
                        conversion_wake.set()
                        await asyncio.sleep(1.0)
                    continue

                done, _ = await asyncio.wait(pending_downloads, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    row = pending_downloads.pop(task)
                    try:
                        path, size = task.result()
                    except Exception as exc:
                        await state.mark_media_failed(row["media_key"], str(exc))
                        continue
                    await state.mark_media_complete(row["media_key"], path, size)
                    if _is_tiff_conversion_candidate(row, path):
                        await state.enqueue_media_conversion(row["media_key"], path)
                        conversion_wake.set()
        except BaseException as exc:
            download_error = exc
        finally:
            downloads_done.set()
            conversion_wake.set()
            try:
                await downloader.close()
            except BaseException as exc:
                close_error = exc
        if conversion_task is not None:
            try:
                await conversion_task
            except BaseException as exc:
                conversion_error = exc
        if download_error is not None:
            raise download_error
        if close_error is not None:
            raise close_error
        if conversion_error is not None:
            raise conversion_error
    finally:
        if conversion_task is not None and not conversion_task.done():
            conversion_task.cancel()
            await asyncio.gather(conversion_task, return_exceptions=True)
        if existing_state is None:
            await state.close()


async def _convert_media(config: ScraperConfig, existing_state: StateStore | None = None) -> None:
    if not _conversion_enabled(config):
        raise ValueError("CJXL_PATH and EXIFTOOL_PATH, or --cjxl-path and --exiftool-path, are required for convert-media")

    state = existing_state or StateStore(config.database_path)
    if existing_state is None:
        await state.initialize()
    try:
        await _run_conversion_worker(config, state)
    finally:
        if existing_state is None:
            await state.close()


async def _run_conversion_worker(
    config: ScraperConfig,
    state: StateStore,
    *,
    wake_event: asyncio.Event | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    if not _conversion_enabled(config):
        return

    converter = JxlConverter(
        cjxl_path=config.cjxl_path,
        exiftool_path=config.exiftool_path,
        store=RecordStore(config.output_dir),
        max_pixels=config.conversion_max_pixels,
        quality=config.jxl_quality,
        timeout=config.conversion_timeout,
        progress=_print_progress,
    )
    timeout_text = f"timeout={config.conversion_timeout}s" if config.conversion_timeout is not None else "timeout=none"
    _print_progress(
        f"conversion worker ready: workers={config.conversion_workers}, quality={config.jxl_quality}, "
        f"max_pixels={config.conversion_max_pixels}, {timeout_text}"
    )
    await state.enqueue_pending_tiff_conversions()
    failed_this_run: set[tuple[str, str]] = set()
    while True:
        rows = await state.next_conversion_batch(
            config.conversion_workers,
            max_attempts=config.retry_limit,
            exclude=failed_this_run,
        )
        if not rows:
            if stop_event is None or stop_event.is_set():
                return
            if wake_event is None:
                await asyncio.sleep(1.0)
            else:
                await wake_event.wait()
                wake_event.clear()
            continue
        results = await asyncio.gather(*(converter.convert_row(row) for row in rows), return_exceptions=True)
        for row, result in zip(rows, results, strict=True):
            if isinstance(result, Exception):
                media_key = row["media_key"]
                target_format = row["target_format"]
                attempt = int(row["attempts"]) + 1
                _print_progress(
                    f"conversion failed: {media_key} ({target_format}) attempt {attempt}/{config.retry_limit}: {result}"
                )
                await state.mark_conversion_failed(
                    media_key,
                    str(result),
                    target_format=target_format,
                )
                await state.record_failure(
                    source="jxl-conversion",
                    identifier=media_key,
                    payload={"source_path": row["source_path"], "target_format": target_format},
                    error=str(result),
                )
                failed_this_run.add((media_key, target_format))
            else:
                path, size = result
                await state.mark_conversion_complete(
                    row["media_key"],
                    path,
                    size,
                    target_format=row["target_format"],
                )
                _delete_source_tiff(Path(row["source_path"]))


async def _status(config: ScraperConfig) -> None:
    state = StateStore(config.database_path)
    await state.initialize()
    try:
        print(json.dumps(await state.status_counts(), indent=2, sort_keys=True))
    finally:
        await state.close()


def _config_from_args(args: argparse.Namespace) -> ScraperConfig:
    include_units = tuple(_split_csv(args.include_units))
    exclude_units = tuple(_split_csv(args.exclude_units))
    return build_config(
        api_key=args.api_key,
        output_dir=args.output_dir,
        database_path=args.database_path,
        metadata_workers=args.metadata_workers,
        media_workers=args.media_workers,
        http_connections=args.http_connections,
        rows_per_page=args.rows,
        record_limit=args.limit,
        retry_limit=args.retry_limit,
        request_timeout=args.timeout,
        conversion_workers=args.conversion_workers,
        conversion_backlog_limit=args.conversion_backlog_limit,
        conversion_timeout=args.conversion_timeout,
        conversion_max_pixels=args.conversion_max_pixels,
        jxl_quality=args.jxl_quality,
        cjxl_path=args.cjxl_path,
        exiftool_path=args.exiftool_path,
        query=args.query,
        sort=args.sort,
        record_type=args.record_type,
        row_group=args.row_group,
        include_units=include_units,
        exclude_units=exclude_units,
        download_media=args.download_media,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smithsonian-scraper")
    parser.add_argument("command", choices=["probe", "scrape", "resume", "download-media", "convert-media", "status"])
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--database-path", default=None)
    parser.add_argument("--metadata-workers", type=int, default=4)
    parser.add_argument("--media-workers", type=int, default=2)
    parser.add_argument("--http-connections", type=int, default=6)
    parser.add_argument("--rows", type=int, default=1000, help="API page size, not a total record limit. Max 1000.")
    parser.add_argument("--limit", "--max-records", dest="limit", type=int, default=None, help="Maximum records to process in this run.")
    parser.add_argument("--retry-limit", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--conversion-workers", type=int, default=None)
    parser.add_argument("--conversion-backlog-limit", type=int, default=None)
    parser.add_argument("--conversion-timeout", type=float, default=None)
    parser.add_argument("--conversion-max-pixels", type=int, default=None)
    parser.add_argument("--jxl-quality", type=int, default=None)
    parser.add_argument("--cjxl-path", default=None)
    parser.add_argument("--exiftool-path", default=None)
    parser.add_argument("--query", default="*:*")
    parser.add_argument("--sort", default="id", choices=["id", "newest", "updated", "random"])
    parser.add_argument("--record-type", default="edanmdm")
    parser.add_argument("--row-group", default="objects", choices=["objects", "archives"])
    parser.add_argument("--include-units", default="")
    parser.add_argument("--exclude-units", default="")
    parser.add_argument("--download-media", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _is_tiff_conversion_candidate(row, path) -> bool:
    return str(row["kind"]) == "highres_tiff" and path.suffix.lower() in {".tif", ".tiff"}


def _require_api_key(config: ScraperConfig) -> None:
    if not config.api_key:
        raise ValueError("SMITHSONIAN_API_KEY or --api-key is required")


def _conversion_enabled(config: ScraperConfig) -> bool:
    return config.cjxl_path is not None and config.exiftool_path is not None


def _delete_source_tiff(source_path: Path) -> None:
    if source_path.suffix.lower() in {".tif", ".tiff"} and source_path.exists():
        source_path.unlink()


def _print_progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())