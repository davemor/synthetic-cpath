"""Verify that the selected source and aggregate inputs match their frozen hashes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
records = json.loads((root / "SOURCE_MANIFEST.json").read_text())
failures = []
for item in records:
    path = root / item["file"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
        failures.append(item["file"])
if failures:
    raise SystemExit("Manifest mismatch: " + ", ".join(failures))
print(f"Verified {len(records)} selected source/input files")
