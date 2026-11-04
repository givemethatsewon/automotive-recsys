"""Stage source + supplied data for Colab; never include .venv or credentials."""
import hashlib
import json
from pathlib import Path
import tarfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bundle():
    root = Path(__file__).resolve().parents[1]
    staging = root / "results" / "colab" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    files = [(path, path.relative_to(root).as_posix()) for path in sorted((root / "src").rglob("*.py"))]
    files += [(path, path.relative_to(root).as_posix()) for path in sorted((root / "tests").glob("*.py"))]
    files += [(path, path.relative_to(root).as_posix()) for path in sorted((root / "scripts").iterdir()) if path.suffix in {".py", ".sh"}]
    files += [(root / name, name) for name in ["pyproject.toml", "uv.lock", "AGENTS.md"]]
    files += [(path, path.relative_to(root).as_posix()) for path in sorted((root / "docs").glob("*.md"))]
    files += [(path, f"data/{path.name}") for path in sorted((root / "data").iterdir()) if path.suffix in {".parquet", ".json", ".txt"}]
    files += [(root / "results/popularity_valid/metrics.json", "reference/popularity_valid_metrics.json")]
    archive = staging / "input.tar"
    with tarfile.open(archive, "w") as bundle:
        for source, name in files:
            bundle.add(source, arcname=name, recursive=False)
    manifest = {"archive_sha256": sha256(archive), "files": {name: sha256(path) for path, name in files}, "parts": []}
    with archive.open("rb") as stream:
        index = 0
        while block := stream.read(16 << 20):
            part = staging / f"input.part{index:03d}"
            part.write_bytes(block)
            manifest["parts"].append({"name": part.name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
            index += 1
    (staging / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive.unlink()
    print(json.dumps({"parts": len(manifest["parts"]), "bytes": sum(part["bytes"] for part in manifest["parts"]), "files": len(files)}, indent=2), flush=True)
    return staging, manifest


if __name__ == "__main__":
    build_bundle()
