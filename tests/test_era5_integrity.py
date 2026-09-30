"""Copy verifier must reject missing files and even same-size corruption."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts import check_era5_integrity as checker


def make_archive(tmp_path, names=("static.nc",)):
    root = tmp_path / "archive"
    root.mkdir()
    rows = []
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"original data bytes"
        path.write_bytes(payload)
        sidecar = b'{"size": 19, "note": "reference metadata"}\n'
        path.with_suffix(".json").write_bytes(sidecar)
        kind = "static" if name == "static.nc" else Path(name).stem
        rows.append({"path": name, "kind": kind, "size": len(payload),
                     "sha256": hashlib.sha256(payload).hexdigest(),
                     "sidecar_sha256": hashlib.sha256(sidecar).hexdigest()})
    manifest = {"schema_version": 1, "files": rows, "expected_files": len(rows),
                "total_bytes": sum(row["size"] for row in rows)}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, checker.load_manifest(manifest_path)


def test_every_expected_file_is_hashed_and_progress_counts_bytes(tmp_path):
    root, manifest = make_archive(tmp_path, ("static.nc", "20220127/surface.nc"))
    blocks = []
    report = checker.inspect_archive(root, manifest, progress=lambda i, name, count: blocks.append(count))
    assert report["status"] == "PASS"
    assert report["all_data_sha256_verified"]
    assert report["counts"] == {"SHA256_OK": 2}
    assert sum(blocks) == manifest["total_bytes"]


@pytest.mark.parametrize("damage,expected_status", [
    ("missing", "MISSING"), ("truncated", "SIZE_MISMATCH"),
    ("empty", "SIZE_MISMATCH"), ("same_size", "HASH_MISMATCH"),
])
def test_damaged_copy_never_passes(tmp_path, damage, expected_status):
    root, manifest = make_archive(tmp_path)
    path = root / "static.nc"
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(b"partial")
    elif damage == "empty":
        path.write_bytes(b"")
    else:
        path.write_bytes(b"X" + path.read_bytes()[1:])
    report = checker.inspect_archive(root, manifest)
    assert report["status"] == "FAIL"
    assert not report["all_data_sha256_verified"]
    assert report["counts"] == {expected_status: 1}


def test_empty_wrong_directory_reports_all_expected_files_missing(tmp_path):
    _, manifest = make_archive(tmp_path)
    empty = tmp_path / "wrong"
    empty.mkdir()
    report = checker.inspect_archive(checker.choose_root(empty), manifest)
    assert report["status"] == "FAIL"
    assert report["counts"]["MISSING"] == manifest["expected_files"]


def test_missing_sidecar_does_not_prevent_independent_data_verification(tmp_path):
    root, manifest = make_archive(tmp_path)
    (root / "static.json").unlink()
    report = checker.inspect_archive(root, manifest)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["all_data_sha256_verified"]
    assert report["issue_counts"] == {"MISSING_SIDECAR": 1}


def test_changed_sidecar_is_reported_even_when_data_matches(tmp_path):
    root, manifest = make_archive(tmp_path)
    (root / "static.json").write_bytes(b"changed metadata")
    report = checker.inspect_archive(root, manifest)
    assert report["status"] == "FAIL"
    assert report["all_data_sha256_verified"]
    assert report["issue_counts"] == {"SIDECAR_MISMATCH": 1}


def test_quick_mode_cannot_claim_content_integrity(tmp_path):
    root, manifest = make_archive(tmp_path)
    path = root / "static.nc"
    path.write_bytes(b"X" + path.read_bytes()[1:])
    report = checker.inspect_archive(root, manifest, quick=True)
    assert report["status"] == "QUICK_CHECK_ONLY"
    assert not report["all_data_sha256_verified"]


def test_unreadable_file_is_an_error(tmp_path, monkeypatch):
    root, manifest = make_archive(tmp_path)
    original_digest = checker.file_digest

    def fail_data_only(path, on_block=None):
        if path.suffix == ".nc":
            raise PermissionError("disk read failed")
        return original_digest(path, on_block)

    monkeypatch.setattr(checker, "file_digest", fail_data_only)
    report = checker.inspect_archive(root, manifest)
    assert report["status"] == "FAIL"
    assert report["issue_counts"] == {"READ_ERROR": 1}


def test_interruption_saves_partial_status_without_claiming_success(tmp_path, monkeypatch):
    root, manifest = make_archive(tmp_path)

    def interrupt(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(checker, "file_digest", interrupt)
    report = checker.inspect_archive(root, manifest)
    assert report["status"] == "INCOMPLETE"
    assert not report["all_data_sha256_verified"]
    checker.write_reports(report, tmp_path / "reports")
    assert json.loads((tmp_path / "reports/report.json").read_text())["interrupted"]


def test_file_changed_during_hash_is_rejected(tmp_path):
    path = tmp_path / "data.nc"
    path.write_bytes(b"original bytes")
    changed = False

    def mutate_file(_):
        nonlocal changed
        if not changed:
            changed = True
            with path.open("ab") as handle:
                handle.write(b"extra byte")

    with pytest.raises(OSError, match="changed"):
        checker.file_digest(path, mutate_file)


def test_extra_files_are_only_reported_and_never_deleted(tmp_path):
    root, manifest = make_archive(tmp_path)
    for name in ("extra.nc", "upper.nc.download.bad-old"):
        (root / name).write_bytes(b"keep me")
    report = checker.inspect_archive(root, manifest)
    assert report["status"] == "PASS_WITH_WARNINGS"
    assert report["issue_counts"] == {"EXTRA_NETCDF": 1, "DOWNLOAD_LEFTOVER": 1}
    assert (root / "extra.nc").read_bytes() == b"keep me"
    assert (root / "upper.nc.download.bad-old").read_bytes() == b"keep me"


@pytest.mark.parametrize("name", ["../static.nc", "/static.nc", "F:/static.nc", "20220127\\surface.nc"])
def test_manifest_rejects_paths_outside_archive(tmp_path, name):
    _, manifest = make_archive(tmp_path)
    manifest["files"][0]["path"] = name
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="relative path"):
        checker.load_manifest(path)


def test_manifest_detects_modification_and_duplicate_entries(tmp_path):
    _, manifest = make_archive(tmp_path)
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum mismatch"):
        checker.load_manifest(path, "0" * 64)
    manifest["files"].append(manifest["files"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        checker.load_manifest(path)


def test_drive_root_detects_nested_archive(tmp_path):
    root, _ = make_archive(tmp_path)
    destination = tmp_path / "era5_inputs"
    root.rename(destination)
    assert checker.choose_root(tmp_path) == destination


def test_missing_drive_is_not_reported_as_data_success(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        checker.choose_root(tmp_path / "missing")


def test_bundled_manifest_covers_the_frozen_archive():
    manifest = checker.load_manifest(checker.DEFAULT_MANIFEST, checker.MANIFEST_SHA256)
    assert manifest["expected_files"] == 1207
    assert manifest["total_bytes"] == 171098308501
    assert manifest["source_values_verified_files"] == 1207
    assert manifest["by_kind"] == {"surface": 397, "upper": 397, "precipitation": 412, "static": 1}
