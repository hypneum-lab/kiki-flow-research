import hashlib
import json
from pathlib import Path

import kiki_flow_core

WEIGHTS_DIR = Path(kiki_flow_core.__file__).parent / "track3_deploy" / "weights"
MANIFEST_PATH = WEIGHTS_DIR / "MANIFEST.json"


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_schema_version_and_latest() -> None:
    manifest = _load_manifest()
    assert manifest["schema_version"] == "1.0"
    assert manifest["latest"] in manifest["files"]


def test_manifest_entries_match_shipped_files() -> None:
    manifest = _load_manifest()
    for filename, meta in manifest["files"].items():
        path = WEIGHTS_DIR / filename
        assert path.is_file(), f"Missing weight file: {filename}"
        data = path.read_bytes()
        assert len(data) == meta["bytes"]
        assert hashlib.sha256(data).hexdigest() == meta["sha256"]


def test_all_safetensors_files_are_listed_in_manifest() -> None:
    manifest = _load_manifest()
    on_disk = {p.name for p in WEIGHTS_DIR.glob("*.safetensors")}
    assert on_disk == set(manifest["files"].keys())
