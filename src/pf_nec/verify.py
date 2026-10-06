"""Verify the public payload against its deterministic build manifest."""
import hashlib
import json
from .config import EXPORT_ROOT


def main():
    manifest = json.loads((EXPORT_ROOT / "MANIFEST.json").read_text())
    for record in manifest["files"]:
        path = EXPORT_ROOT / record["path"]
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Export integrity mismatch: " + record["path"])
    print(json.dumps({"integrity": "PASS", "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
