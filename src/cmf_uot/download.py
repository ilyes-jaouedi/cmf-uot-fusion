"""Download the acoustic-camera dataset from Zenodo into data/ (or $CMF_UOT_DATA).

    poetry run python -m cmf_uot.download                 # whole dataset (7.1 GB)
    poetry run python -m cmf_uot.download --paper-only    # only what the article uses (0.5 GB)

Zenodo records are flat; the files are placed back into waves/, camera_detections/
and images/ using folders.json, and every file is checked against checksums.sha256.
Files already present with the right checksum are skipped.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from .paths import DATA_DIR

ZENODO_RECORD = None  # set to the record id once the dataset is published, e.g. "1234567"

# recordings used in the article: configurations 1 (standard), 2 (acoustic ambiguity), 4 (occlusion)
PAPER_FILES = {"array.npy", "mesure_manip1_all.npy", "mesure_manip2_all.npy", "mesure_manip4_all.npy"}


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def _sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", default=ZENODO_RECORD, help="Zenodo record id")
    ap.add_argument("--dest", default=str(DATA_DIR), help="destination folder (default: data/)")
    ap.add_argument("--paper-only", action="store_true", help="skip recordings not used in the article")
    args = ap.parse_args(argv)
    if not args.record:
        sys.exit("No Zenodo record id yet: pass --record <id>.")

    record = json.loads(_get(f"https://zenodo.org/api/records/{args.record}"))
    links = {f["key"]: f["links"]["self"] for f in record["files"]}
    folders = json.loads(_get(links["folders.json"]))
    sums = dict(line.split()[::-1] for line in _get(links["checksums.sha256"]).decode().splitlines() if line)

    dest = Path(args.dest)
    for name, url in sorted(links.items()):
        if name in ("folders.json", "checksums.sha256"):
            continue
        if args.paper_only and name.endswith(".npy") and folders.get(name) == "waves" and name not in PAPER_FILES:
            continue
        target = dest / folders.get(name, ".") / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and name in sums and _sha256(target) == sums[name]:
            print(f"ok      {target.relative_to(dest)}")
            continue
        print(f"get     {target.relative_to(dest)}", flush=True)
        urllib.request.urlretrieve(url, target)
        if name in sums and _sha256(target) != sums[name]:
            target.unlink()
            sys.exit(f"checksum mismatch for {name}; file removed, please retry")
    print(f"Dataset ready in {dest}")


if __name__ == "__main__":
    main()
