"""Pinned downloads with checksum verification (demo data and the benchmark)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import httpx

from quarry.errors import DatasetError


@dataclass(frozen=True)
class RemoteFile:
    """A file at a fixed URL, pinned by SHA-256.

    ``sha256=None`` means trust on first use: the size must match, and the hash of the
    first verified download is written next to the file (``<name>.sha256``) and enforced
    from then on. Only the Spider archive uses it (see ``quarry.eval.spider``).
    """

    name: str
    url: str
    sha256: str | None
    size_bytes: int
    source_page: str


HILLSTROM = RemoteFile(
    name="hillstrom.csv",
    url=(
        "http://www.minethatdata.com/"
        "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    ),
    sha256="0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece",
    size_bytes=3_964_977,
    source_page="https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html",
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin_path(path: Path) -> Path:
    return path.with_name(path.name.removesuffix(".part") + ".sha256")


def verify(path: Path, remote: RemoteFile) -> None:
    """Check the pinned hash (or, for trust-on-first-use files, size then recorded hash)."""
    actual = sha256_of(path)
    expected = remote.sha256
    if expected is None:
        size = path.stat().st_size
        if size != remote.size_bytes:
            raise DatasetError(
                f"{path} is {size:,} bytes, expected {remote.size_bytes:,}. Delete it and "
                f"download again, or fetch it by hand from {remote.source_page}."
            )
        pin = _pin_path(path)
        if not pin.exists():
            pin.write_text(f"{actual}\n", encoding="utf-8")
        expected = pin.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise DatasetError(
            f"{path} has SHA-256 {actual}, expected {expected}. Delete it and download "
            f"again, or fetch it by hand from {remote.source_page}."
        )


def fetch(remote: RemoteFile, directory: Path, *, timeout_s: float = 600.0) -> Path:
    """Download ``remote`` into ``directory`` unless a verified copy is already there."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / remote.name
    if target.exists():
        verify(target, remote)
        return target
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with (
            httpx.stream("GET", remote.url, follow_redirects=True, timeout=timeout_s) as response,
            partial.open("wb") as handle,
        ):
            response.raise_for_status()
            kind = response.headers.get("content-type", "")
            if "text/html" in kind:
                raise DatasetError(
                    f"{remote.url} returned an HTML page instead of the file (for Google Drive "
                    f"this usually means the daily download quota is exhausted). Retry later "
                    f"or download it by hand from {remote.source_page} into {target}."
                )
            for chunk in response.iter_bytes(1 << 20):
                handle.write(chunk)
    except httpx.HTTPError as exc:
        partial.unlink(missing_ok=True)
        raise DatasetError(f"download of {remote.url} failed: {exc}") from exc
    except DatasetError:
        partial.unlink(missing_ok=True)
        raise
    try:
        verify(partial, remote)
    except DatasetError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(target)
    return target
