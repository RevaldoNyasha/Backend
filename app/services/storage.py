"""Where uploaded files live.

The rest of the app only uses storage *keys* (e.g. ``"{project_id}/{uuid}.pdf"``,
saved as ``Document.storage_path``) and the ``Storage`` interface below. Today the
files sit on the local disk under ``uploads/``; moving to the Oracle VM means
adding a second implementation and returning it from ``get_storage()``.
"""

import shutil
from pathlib import Path, PurePosixPath
from typing import Protocol

from app.core.config import UPLOADS_ROOT


class Storage(Protocol):
    def save(self, key: str, content: bytes) -> None: ...

    def read(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...

    def delete(self, key: str) -> None:
        """Remove one file; missing files are ignored."""
        ...

    def delete_prefix(self, prefix: str) -> None:
        """Remove every file under a prefix (e.g. a whole project); missing is fine."""
        ...


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _path(self, key: str) -> Path:
        # Keys are app-generated, but refuse anything that could escape the root.
        # Backslashes and ":" matter on Windows (`..\x`, `C:x`) even though
        # PurePosixPath does not split on them.
        parts = PurePosixPath(key).parts
        if (
            not parts
            or key.startswith("/")
            or "\\" in key
            or ":" in key
            or any(part == ".." for part in parts)
        ):
            raise ValueError(f"Invalid storage key: {key!r}")
        return self.root.joinpath(*parts)

    def save(self, key: str, content: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def read(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self._path(prefix), ignore_errors=True)


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = LocalStorage(UPLOADS_ROOT)
    return _storage


def set_storage(storage: Storage | None) -> None:
    """Swap the storage backend (tests use a temporary folder)."""
    global _storage
    _storage = storage
