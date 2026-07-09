#!/usr/bin/env python
"""Encode BrainNet's canonical .txt subject splits into an HCP-style split JSON.

BrainNet ships splits as bare subject-ID lists at
`/projects/brainnet-data/subject_splits/<DS>.<split>.txt` (one `sub-XXXX` per
line). This converts them into the `{split: [{"img", "seg4"}]}` JSON that
`validate_inr_brainnet.py` / the dataloaders already consume — so no reader code
has to learn the txt format. Excluded subjects (`<DS>.exclude.txt` and the
dataset's entry in `bad_surfaces.json`) are dropped at conversion time.

    python build_split_json.py --dataset ABIDE
    python build_split_json.py --dataset OASIS3 --out config/OASIS3_split.json
"""

import argparse
import json
from pathlib import Path

SPLIT_DIR = Path("/projects/brainnet-data/subject_splits")
# txt split name -> JSON key. BrainNet names the val split "validation"; keep it.
SPLITS = ("train", "validation", "test")


def excluded_subjects(dataset: str) -> set[str]:
    excluded: set[str] = set()
    exclude_txt = SPLIT_DIR / f"{dataset}.exclude.txt"
    if exclude_txt.exists():
        excluded |= {l.strip() for l in exclude_txt.read_text().splitlines() if l.strip()}
    bad_json = SPLIT_DIR / "bad_surfaces.json"
    if bad_json.exists():
        excluded |= set(json.loads(bad_json.read_text()).get(dataset, []))
    return excluded


def read_ids(dataset: str, split: str) -> list[str]:
    f = SPLIT_DIR / f"{dataset}.{split}.txt"
    if not f.exists():
        return []
    return [l.strip() for l in f.read_text().splitlines() if l.strip()]


def build(dataset: str) -> dict[str, list[dict[str, str]]]:
    excluded = excluded_subjects(dataset)
    out: dict[str, list[dict[str, str]]] = {}
    for split in SPLITS:
        ids = [s for s in read_ids(dataset, split) if s not in excluded]
        if ids:
            out[split] = [
                {"img": f"{s}/t1w.nii.gz", "seg4": f"{s}/seg.nii.gz"} for s in ids
            ]
    if excluded:
        print(f"  dropped {len(excluded)} excluded subject(s)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, help="e.g. ABIDE, OASIS3")
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output JSON (default: config/<DS>_split.json)",
    )
    args = p.parse_args()

    out_path = args.out or Path("config") / f"{args.dataset}_split.json"
    splits = build(args.dataset)
    if not splits:
        raise SystemExit(f"No split .txt files found for {args.dataset} in {SPLIT_DIR}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits, indent=2))
    counts = {k: len(v) for k, v in splits.items()}
    print(f"Wrote {out_path}  ->  {counts}")


if __name__ == "__main__":
    main()
