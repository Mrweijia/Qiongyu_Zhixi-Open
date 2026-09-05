"""Validate the GOV-01 public-release asset manifest against this checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs" / "public_release_assets.json"
FORBIDDEN_MARKERS = (b"F:\\F_Area", b"C:\\Users\\86135")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path = ROOT, manifest_path: Path = MANIFEST) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    approved = manifest["approved_assets"]
    approved_paths = {asset["path"] for asset in approved}
    excluded_paths = {item["path"] for item in manifest["excluded_paths"]}
    if len(approved_paths) != len(approved):
        errors.append("approved asset paths must be unique")
    if not all(item.get("reason") for item in manifest["excluded_paths"]):
        errors.append("every excluded path needs a reason")
    if approved_paths & excluded_paths:
        errors.append("an asset cannot be both approved and excluded")

    for asset in approved:
        path = root / asset["path"]
        if not path.is_file():
            errors.append(f"missing approved asset: {asset['path']}")
            continue
        expected = asset.get("sha256", "")
        if expected and sha256(path) != expected:
            errors.append(f"hash mismatch: {asset['path']}")
        content = path.read_bytes()
        if any(marker in content for marker in FORBIDDEN_MARKERS):
            errors.append(f"personal path found in approved asset: {asset['path']}")
        if b"__pycache__" in content:
            errors.append(f"cache marker found in approved asset: {asset['path']}")
    templates = manifest["release_notice_template"]["files"]
    expected_text = {
        "code_and_weight_license": b"Apache License\n",
        "data_license": b"Creative Commons Attribution 4.0 International Public License",
        "attribution_and_exclusions": b"Source: NOAA National Centers for Environmental Information (NCEI), Integrated Surface Database.",
    }
    for label, marker in expected_text.items():
        template = root / templates[label]
        if not template.is_file() or marker not in template.read_bytes():
            errors.append(f"invalid release template: {label}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()
    errors = validate(args.root.resolve(), args.manifest.resolve())
    if errors:
        print("public-release asset validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("public-release asset manifest is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
