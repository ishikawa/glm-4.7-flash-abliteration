"""Release validation and CLI plans with no network or real subprocesses."""

import hashlib
import subprocess

import hf_publish as publish
import pytest


@pytest.fixture
def release(tmp_path, monkeypatch):
    directory = tmp_path / "hf"
    directory.mkdir()
    payload = b"synthetic GGUF payload"
    digest = hashlib.sha256(payload).hexdigest()
    (directory / "README.md").write_text(
        f"| `{publish.FILENAME}` | {len(payload)} | `{digest}` |\n"
    )
    for name in ("LICENSE", "NOTICE"):
        (directory / name).write_text(name)
    gguf = tmp_path / "model with spaces.gguf"
    gguf.write_bytes(payload)
    monkeypatch.setattr(publish, "HF_DIR", directory)
    return gguf, directory


def test_dry_run_does_not_invoke_cli_or_expose_token(release, monkeypatch, capsys):
    gguf, _ = release
    monkeypatch.setenv("HF_TOKEN", "synthetic-secret-do-not-print")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("CLI invoked"))
    assert publish.main([str(gguf)]) == 0
    output = capsys.readouterr().out
    assert "dry-run" in output and "verified" in output
    assert publish.DEFAULT_REPO in output
    assert "synthetic-secret" not in output
    assert "hf repos create" in output and "hf upload" in output


def test_missing_gguf_dry_run_and_upload(release, monkeypatch, capsys):
    gguf, _ = release
    gguf.unlink()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("CLI invoked"))
    assert publish.main([str(gguf)]) == 0
    assert "not verified" in capsys.readouterr().err
    with pytest.raises(SystemExit) as error:
        publish.main([str(gguf), "--yes"])
    assert error.value.code == 1


@pytest.mark.parametrize("same_size", [False, True])
@pytest.mark.parametrize("yes", [False, True])
def test_bad_payload_rejected_before_any_cli(release, monkeypatch, same_size, yes):
    gguf, _ = release
    gguf.write_bytes(b"x" * gguf.stat().st_size if same_size else b"short")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("CLI invoked"))
    with pytest.raises(SystemExit) as error:
        publish.main([str(gguf), *(["--yes"] if yes else [])])
    assert error.value.code == 1


def test_upload_uses_exact_destinations(release, monkeypatch):
    gguf, directory = release
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    assert publish.main([str(gguf), "--repo-id", "owner/release", "--yes"]) == 0
    assert calls[0][0] == [
        "hf",
        "repos",
        "create",
        "owner/release",
        "--type",
        "model",
        "--exist-ok",
    ]
    assert [c[0][4] for c in calls[1:]] == [publish.FILENAME, *publish.DOCUMENTS]
    assert calls[1][0][3] == str(gguf)
    assert all(kw == {"check": True} for _, kw in calls)
    assert len(calls) == 5
    assert all(str(directory) in c[0][3] for c in calls[2:])


@pytest.mark.parametrize("name", publish.DOCUMENTS)
def test_missing_document_blocks_upload(release, name, monkeypatch):
    gguf, directory = release
    (directory / name).unlink()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("CLI invoked"))
    with pytest.raises(SystemExit):
        publish.main([str(gguf), "--yes"])


def test_cli_failure_stops_remaining_uploads(release, monkeypatch):
    gguf, _ = release
    calls = []

    def fail(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(SystemExit):
        publish.main([str(gguf), "--yes"])
    assert len(calls) == 1


def test_invalid_card_and_repo(release, monkeypatch):
    gguf, directory = release
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("CLI invoked"))
    with pytest.raises(SystemExit):
        publish.main([str(gguf), "--repo-id", "--help"])
    (directory / "README.md").write_text("No checksum")
    with pytest.raises(SystemExit):
        publish.main([str(gguf)])


def test_repository_model_card_metadata():
    size, digest = publish.release_metadata(publish.HF_DIR)
    assert size == 17_520_169_312
    assert digest == "33a80f475f219f86e9a9785c278ef46aa226a9c50a426fc42985bec8b7aa469f"
