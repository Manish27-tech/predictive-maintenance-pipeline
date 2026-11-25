import json

from maintenance.manifest import build_manifest, sha256, write_manifest


def test_manifest_records_checksums(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")
    manifest = build_manifest([first, second], tmp_path)
    assert manifest["file_count"] == 2
    assert manifest["total_bytes"] == 9
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    assert sha256(first) == manifest["files"][0]["sha256"]


def test_manifest_skips_missing_files(tmp_path):
    present = tmp_path / "present.txt"
    present.write_text("data", encoding="utf-8")
    manifest = build_manifest([present, tmp_path / "absent.txt"], tmp_path)
    assert manifest["file_count"] == 1


def test_write_manifest_is_valid_json(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("payload", encoding="utf-8")
    manifest = build_manifest([source], tmp_path, metadata={"project": "demo"})
    path = write_manifest(manifest, tmp_path / "nested" / "manifest.json")
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["metadata"]["project"] == "demo"
    assert restored["files"][0]["path"] == "source.txt"
