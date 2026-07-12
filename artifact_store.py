"""Validated, atomic JSON and text artifact storage."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Mapping, Optional

from job_store import validate_job_id


_ARTIFACT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PRINCIPAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@-]{0,127}$")
OWNER_BINDING_NAME = "_owner.json"


class ArtifactStoreError(RuntimeError):
    """Raised for invalid or failed artifact operations."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_owner_id(owner_id: str) -> str:
    value = str(owner_id or "").strip().lower()
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ArtifactStoreError("artifact owner_id must be a UUID") from exc
    if value not in {parsed.hex, str(parsed)}:
        raise ArtifactStoreError("artifact owner_id must be a compact or canonical UUID")
    return value


def _validate_principal_id(principal_id: str) -> str:
    value = str(principal_id or "").strip()
    if not _PRINCIPAL_ID_RE.fullmatch(value):
        raise ArtifactStoreError("artifact principal_id contains unsupported characters")
    return value


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


class ArtifactStore:
    def __init__(self, root: Optional[str | Path] = None) -> None:
        configured = root if root is not None else os.getenv("ARTIFACT_DIR", "./artifacts")
        self.root = Path(configured).expanduser().resolve()

    def path_for(self, job_id: str, name: str = "result") -> Path:
        job_id_value = validate_job_id(job_id)
        return self._owner_path(job_id_value, name, ".json")

    def _owner_path(self, owner_id: str, name: str, suffix: str) -> Path:
        name_value = str(name or "").strip().lower()
        if not _ARTIFACT_NAME_RE.fullmatch(name_value):
            raise ArtifactStoreError(
                "artifact name must use lowercase letters, digits, underscores, and hyphens"
            )
        return self._safe_path(Path(owner_id) / f"{name_value}{suffix}")

    def resolve_relative_path(self, relative_path: str | Path) -> Path:
        raw_path = os.fspath(relative_path)
        if not isinstance(raw_path, str):
            raise ArtifactStoreError("artifact path must be text")
        portable_parts = raw_path.replace("\\", "/").split("/")
        if any(part in {"", ".", ".."} or ":" in part for part in portable_parts):
            raise ArtifactStoreError("artifact path contains a disallowed segment")
        relative = Path(relative_path)
        if relative.is_absolute() or not relative.parts:
            raise ArtifactStoreError("artifact path must be relative to ARTIFACT_DIR")
        return self._safe_path(relative)

    def canonical_relative_path(self, relative_path: str | Path) -> str:
        return self.resolve_relative_path(relative_path).relative_to(self.root).as_posix()

    async def bind_owner_principal(self, owner_id: str, principal_id: str) -> dict[str, str]:
        """Persist an immutable artifact-owner binding for authorization checks."""
        owner_id_value = _validate_owner_id(owner_id)
        principal_id_value = _validate_principal_id(principal_id)
        target = self._safe_path(Path(owner_id_value) / OWNER_BINDING_NAME)
        encoded = json.dumps(
            {
                "owner_id": owner_id_value,
                "principal_id": principal_id_value,
                "bound_at": _utc_now_iso(),
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

        await asyncio.to_thread(
            self._create_owner_binding,
            target,
            encoded,
            owner_id_value,
            principal_id_value,
        )
        return {"owner_id": owner_id_value, "principal_id": principal_id_value}

    async def owner_principal(self, owner_id: str) -> Optional[str]:
        owner_id_value = _validate_owner_id(owner_id)
        target = self._safe_path(Path(owner_id_value) / OWNER_BINDING_NAME)

        def read() -> Optional[str]:
            if not target.exists():
                return None
            if _is_link_like(target) or not target.is_file():
                raise ArtifactStoreError("artifact owner binding is invalid")
            try:
                raw = target.read_bytes()
                if len(raw) > 4096:
                    raise ArtifactStoreError("artifact owner binding exceeds its size limit")
                payload = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ArtifactStoreError("artifact owner binding is unreadable") from exc
            if not isinstance(payload, dict) or payload.get("owner_id") != owner_id_value:
                raise ArtifactStoreError("artifact owner binding is invalid")
            return _validate_principal_id(payload.get("principal_id"))

        return await asyncio.to_thread(read)

    async def write_json(self, job_id: str, value: Any, name: str = "result") -> dict[str, Any]:
        name_value = str(name or "").strip().lower()
        target = self.path_for(job_id, name_value)
        try:
            encoded = json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ArtifactStoreError(f"artifact value is not JSON serializable: {exc}") from exc

        await asyncio.to_thread(self._atomic_write, target, encoded)
        relative_path = target.relative_to(self.root).as_posix()
        return {
            "artifact_id": f"{validate_job_id(job_id)}:{name_value}",
            "job_id": validate_job_id(job_id),
            "name": name_value,
            "relative_path": relative_path,
            "path": relative_path,
            "content_type": "application/json",
            "size_bytes": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "created_at": _utc_now_iso(),
        }

    async def write_text(
        self,
        owner_id: str,
        text: str,
        name: str = "content",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        owner_id_value = _validate_owner_id(owner_id)
        if not isinstance(text, str):
            raise ArtifactStoreError("text artifact value must be a string")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ArtifactStoreError("artifact metadata must be a mapping or null")

        metadata_value = None
        if metadata:
            try:
                metadata_value = json.loads(
                    json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)
                )
            except (TypeError, ValueError) as exc:
                raise ArtifactStoreError(f"artifact metadata is not JSON serializable: {exc}") from exc

        name_value = str(name or "").strip().lower()
        target = self._owner_path(owner_id_value, name_value, ".txt")
        encoded = text.encode("utf-8")
        await asyncio.to_thread(self._atomic_write, target, encoded)
        relative_path = target.relative_to(self.root).as_posix()
        result = {
            "artifact_id": f"{owner_id_value}:{name_value}",
            "owner_id": owner_id_value,
            "name": name_value,
            "relative_path": relative_path,
            "path": relative_path,
            "content_type": "text/plain; charset=utf-8",
            "size_bytes": len(encoded),
            "character_count": len(text),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "created_at": _utc_now_iso(),
        }
        if metadata_value is not None:
            result["metadata"] = metadata_value
        return result

    async def read_json(self, relative_path: str | Path) -> Any:
        path = self.resolve_relative_path(relative_path)
        try:
            raw = await asyncio.to_thread(path.read_text, encoding="utf-8")
            return json.loads(raw)
        except FileNotFoundError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactStoreError(f"could not read artifact: {exc}") from exc

    async def read_text(
        self,
        relative_path: str | Path,
        max_chars: Optional[int] = None,
    ) -> str:
        path = self.resolve_relative_path(relative_path)
        if max_chars is not None and (isinstance(max_chars, bool) or not isinstance(max_chars, int)):
            raise ArtifactStoreError("max_chars must be an integer or null")
        if max_chars is not None and max_chars < 0:
            raise ArtifactStoreError("max_chars must be non-negative")

        def read() -> str:
            with path.open("r", encoding="utf-8") as handle:
                return handle.read() if max_chars is None else handle.read(max_chars)

        try:
            return await asyncio.to_thread(read)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ArtifactStoreError(f"could not read artifact: {exc}") from exc

    async def exists(self, relative_path: str | Path) -> bool:
        path = self.resolve_relative_path(relative_path)
        return await asyncio.to_thread(path.is_file)

    async def delete_job_artifacts(self, job_id: str) -> int:
        job_dir = self._safe_path(Path(validate_job_id(job_id)))

        def remove() -> int:
            if not job_dir.exists():
                return 0
            count = 0
            for child in job_dir.iterdir():
                if child.is_file():
                    child.unlink()
                    count += 1
            try:
                job_dir.rmdir()
            except OSError:
                pass
            return count

        return await asyncio.to_thread(remove)

    async def prune_older_than(
        self,
        max_age_seconds: int,
        *,
        protected_owner_ids: Optional[Collection[str]] = None,
    ) -> int:
        """Delete UUID-owned artifact directories older than the retention window."""
        if isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, int):
            raise ArtifactStoreError("max_age_seconds must be an integer")
        if max_age_seconds <= 0:
            return 0
        if protected_owner_ids is None:
            protected_owner_ids = ()
        if isinstance(protected_owner_ids, (str, bytes)):
            raise ArtifactStoreError("protected_owner_ids must be a collection of UUIDs")
        try:
            protected_owner_uuids = {
                uuid.UUID(_validate_owner_id(owner_id)).hex for owner_id in protected_owner_ids
            }
        except TypeError as exc:
            raise ArtifactStoreError("protected_owner_ids must be a collection of UUIDs") from exc

        def prune() -> int:
            if not self.root.exists():
                return 0
            cutoff = time.time() - max_age_seconds
            deleted = 0
            for owner_dir in self.root.iterdir():
                try:
                    if _is_link_like(owner_dir) or not owner_dir.is_dir():
                        continue
                    owner_uuid = uuid.UUID(_validate_owner_id(owner_dir.name)).hex
                    if owner_uuid in protected_owner_uuids:
                        continue
                    entries = list(owner_dir.iterdir())
                    if any(_is_link_like(item) or not item.is_file() for item in entries):
                        continue
                    newest_mtime = max(
                        (item.stat().st_mtime for item in entries),
                        default=owner_dir.stat().st_mtime,
                    )
                    if newest_mtime >= cutoff:
                        continue
                    for item in entries:
                        item.unlink()
                        deleted += 1
                    owner_dir.rmdir()
                except (ArtifactStoreError, FileNotFoundError, OSError):
                    continue
            return deleted

        return await asyncio.to_thread(prune)

    def _safe_path(self, relative: Path) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ArtifactStoreError("artifact path escapes ARTIFACT_DIR") from exc
        return candidate

    @staticmethod
    def _atomic_write(target: Path, encoded: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            fd, raw_temp_path = tempfile.mkstemp(
                dir=target.parent,
                prefix=f".{target.stem}.",
                suffix=".tmp",
            )
            temp_path = Path(raw_temp_path)
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    @staticmethod
    def _create_owner_binding(
        target: Path,
        encoded: bytes,
        owner_id: str,
        principal_id: str,
    ) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None
        try:
            fd, raw_temp_path = tempfile.mkstemp(
                dir=target.parent,
                prefix="._owner.",
                suffix=".tmp",
            )
            temp_path = Path(raw_temp_path)
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temp_path, target)
            except FileExistsError:
                try:
                    existing = json.loads(target.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ArtifactStoreError("artifact owner binding is unreadable") from exc
                if not isinstance(existing, dict) or existing.get("owner_id") != owner_id:
                    raise ArtifactStoreError("artifact owner binding is invalid")
                existing_principal = _validate_principal_id(existing.get("principal_id"))
                if existing_principal != principal_id:
                    raise ArtifactStoreError("artifact owner is already bound to another principal")
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()


_default_artifact_store: Optional[ArtifactStore] = None


def get_artifact_store() -> ArtifactStore:
    global _default_artifact_store
    if _default_artifact_store is None:
        _default_artifact_store = ArtifactStore()
    return _default_artifact_store


async def read_artifact(relative_path: str | Path) -> Any:
    return await get_artifact_store().read_json(relative_path)
