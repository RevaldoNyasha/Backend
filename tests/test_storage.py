from pathlib import Path

import pytest

from app.services.storage import LocalStorage


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(tmp_path / "uploads")


def test_save_read_and_exists(storage: LocalStorage) -> None:
    storage.save("project-1/file.pdf", b"content")

    assert storage.exists("project-1/file.pdf")
    assert storage.read("project-1/file.pdf") == b"content"
    assert (storage.root / "project-1" / "file.pdf").is_file()


def test_delete_removes_one_file_and_ignores_missing(storage: LocalStorage) -> None:
    storage.save("project-1/a.pdf", b"a")
    storage.save("project-1/b.pdf", b"b")

    storage.delete("project-1/a.pdf")
    storage.delete("project-1/never-existed.pdf")

    assert not storage.exists("project-1/a.pdf")
    assert storage.exists("project-1/b.pdf")


def test_delete_prefix_removes_a_whole_project(storage: LocalStorage) -> None:
    storage.save("project-1/a.pdf", b"a")
    storage.save("project-2/b.pdf", b"b")

    storage.delete_prefix("project-1")
    storage.delete_prefix("project-3")

    assert not (storage.root / "project-1").exists()
    assert storage.exists("project-2/b.pdf")


@pytest.mark.parametrize(
    "key",
    [
        "../outside.pdf",
        "project/../../outside.pdf",
        "/etc/passwd",
        "",
        "..\\outside.pdf",
        "project\\..\\..\\outside.pdf",
        "C:outside.pdf",
        "C:/Windows/file.pdf",
    ],
)
def test_keys_cannot_escape_the_root(storage: LocalStorage, key: str) -> None:
    with pytest.raises(ValueError):
        storage.save(key, b"x")
