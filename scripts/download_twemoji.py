import argparse
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


GITHUB_API = "https://api.github.com/repos/jdecked/twemoji"


def _urlopen_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "svg-fusion-twemoji-downloader"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_ref(ref: str) -> str:
    if ref and ref != "latest":
        return ref

    # Try latest release tag first.
    try:
        data = _urlopen_json(f"{GITHUB_API}/releases/latest")
        tag = data.get("tag_name")
        if tag:
            return tag
    except Exception:
        pass

    # Fallback to main branch.
    return "main"


def download_tarball(ref: str, out_path: Path):
    url = f"{GITHUB_API}/tarball/{ref}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "svg-fusion-twemoji-downloader"},
    )
    with urllib.request.urlopen(req) as resp, open(out_path, "wb") as f:
        shutil.copyfileobj(resp, f)


def extract_svgs(tar_path: Path, output_dir: Path, max_files: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with tarfile.open(tar_path, "r:*") as tar:
        members = [m for m in tar.getmembers() if "/assets/svg/" in m.name and m.name.endswith(".svg")]
        members.sort(key=lambda m: m.name)
        if max_files and max_files > 0:
            members = members[:max_files]
        for m in members:
            # Flatten into output_dir with filename only.
            fname = os.path.basename(m.name)
            out_path = output_dir / fname
            with tar.extractfile(m) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Download Twemoji SVG assets")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for SVGs")
    parser.add_argument("--ref", type=str, default="latest", help="Git ref (tag/branch). Use 'latest' for release")
    parser.add_argument("--max_files", type=int, default=0, help="Limit number of SVGs")
    parser.add_argument("--keep_temp", action="store_true", help="Keep downloaded tarball")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    ref = resolve_ref(args.ref)

    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / f"twemoji_{ref}.tar.gz"
        print(f"Downloading Twemoji ({ref}) ...")
        download_tarball(ref, tar_path)
        print("Extracting SVGs ...")
        extracted = extract_svgs(tar_path, output_dir, args.max_files)

        if args.keep_temp:
            keep_path = output_dir / tar_path.name
            shutil.copy2(tar_path, keep_path)
            print(f"Kept tarball: {keep_path}")

    print(f"Done. Extracted {extracted} SVGs to {output_dir}")


if __name__ == "__main__":
    main()
