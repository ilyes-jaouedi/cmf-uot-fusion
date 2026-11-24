"""Download the acoustic-camera dataset from Zenodo into data/ (or $CMF_UOT_DATA).

    poetry run python -m cmf_uot.download                 # whole dataset (2.5 GB to download)
    poetry run python -m cmf_uot.download --paper-only    # only what the article uses (0.2 GB)

Zenodo records are flat; the files are placed back into waves/, camera_detections/
and images/ using folders.json, and every file is checked against checksums.sha256.
The recordings are stored on Zenodo as int32 ADC counts (mesure_*.npz: counts, scale);
each one is turned back into the original float64 .npy (counts * scale, bit-exact) and
checked against original_npy_checksums.sha256, so the experiment scripts read the same
files as in the article (each .npy is 157 MB on disk). Files already present with the
right checksum are skipped.
"""
import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

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
    ap.add_argument("--keep-npz", action="store_true", help="also keep the compressed .npz recordings")
    args = ap.parse_args(argv)
    if not args.record:
        sys.exit("No Zenodo record id yet: pass --record <id>.")

    record = json.loads(_get(f"https://zenodo.org/api/records/{args.record}"))
    links = {f["key"]: f["links"]["self"] for f in record["files"]}
    folders = json.loads(_get(links["folders.json"]))

    def read_sums(name):
        if name not in links:
            return {}
        return dict(line.split()[::-1] for line in _get(links[name]).decode().splitlines() if line)

    sums = read_sums("checksums.sha256")                    # files as stored on Zenodo
    npy_sums = read_sums("original_npy_checksums.sha256")   # original float64 recordings
    meta = {"folders.json", "checksums.sha256", "original_npy_checksums.sha256"}

    dest = Path(args.dest)
    for name, url in sorted(links.items()):
        if name in meta:
            continue
        npy_name = name[:-4] + ".npy" if name.endswith(".npz") else name
        if args.paper_only and folders.get(name) == "waves" and npy_name not in PAPER_FILES:
            continue
        folder = dest / folders.get(name, ".")
        folder.mkdir(parents=True, exist_ok=True)

        if name.endswith(".npz"):
            # recording stored as int32 counts: rebuild the original float64 .npy (bit-exact)
            npy = folder / npy_name
            if npy.exists() and _sha256(npy) == npy_sums.get(npy_name):
                print(f"ok      {npy.relative_to(dest)}")
                continue
            tmp = folder / name
            print(f"get     {npy.relative_to(dest)}", flush=True)
            urllib.request.urlretrieve(url, tmp)
            if name in sums and _sha256(tmp) != sums[name]:
                tmp.unlink()
                sys.exit(f"checksum mismatch for {name}; file removed, please retry")
            with np.load(tmp) as z:
                np.save(npy, z["counts"] * z["scale"])
            if npy_name in npy_sums and _sha256(npy) != npy_sums[npy_name]:
                npy.unlink()
                sys.exit(f"reconstruction of {npy_name} does not match the original; please report it")
            if not args.keep_npz:
                tmp.unlink()
            continue

        target = folder / name
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
