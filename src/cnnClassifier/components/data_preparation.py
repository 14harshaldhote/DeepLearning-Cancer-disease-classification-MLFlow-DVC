"""Leakage-safe train / val / test split.

The raw dataset contains byte-identical copies ("10 - Copy.png") and several
slices per scan ("000005 (3).png"). A random split would put the same scan,
or even the same file, on both sides and inflate the scores. This stage:

1. drops exact duplicates (by content hash),
2. groups files by scan id (the file name before " (n)" / " - Copy"),
3. assigns whole groups to train, val or test.
"""

import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from cnnClassifier import logger
from cnnClassifier.entity.config_entity import DataPreparationConfig

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SPLITS = ("train", "val", "test")


def scan_id(filename: str) -> str:
    """'000005 (3).png' -> '000005', '10 - Copy (2).png' -> '10'."""
    stem = Path(filename).stem
    return re.split(r"\s*(?:\(|-\s*Copy)", stem, maxsplit=1)[0].strip()


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def dedupe(files: list[Path]) -> tuple[list[Path], int]:
    seen, unique = set(), []
    for f in sorted(files):
        h = file_hash(f)
        if h not in seen:
            seen.add(h)
            unique.append(f)
    return unique, len(files) - len(unique)


def group_split(files: list[Path], val: float, test: float, seed: int) -> dict[str, list[Path]]:
    groups = defaultdict(list)
    for f in files:
        groups[scan_id(f.name)].append(f)

    ids = sorted(groups)
    random.Random(seed).shuffle(ids)

    # Fill test, then val, group by group, until each reaches its share of files.
    total = len(files)
    targets = {"test": test * total, "val": val * total}
    split = {name: [] for name in SPLITS}
    for gid in ids:
        for name in ("test", "val"):
            if len(split[name]) < targets[name]:
                split[name].extend(groups[gid])
                break
        else:
            split["train"].extend(groups[gid])
    return split


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config

    def run(self) -> dict:
        src = self.config.source_dir
        out = self.config.root_dir
        for name in SPLITS:
            shutil.rmtree(out / name, ignore_errors=True)

        report = {"classes": {}, "seed": self.config.params_seed}
        for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
            files = [f for f in class_dir.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES]
            unique, dropped = dedupe(files)
            split = group_split(
                unique,
                self.config.params_val_split,
                self.config.params_test_split,
                self.config.params_seed,
            )
            for name, split_files in split.items():
                dest = out / name / class_dir.name
                dest.mkdir(parents=True, exist_ok=True)
                for f in split_files:
                    shutil.copy2(f, dest / f.name)

            report["classes"][class_dir.name] = {
                "raw_files": len(files),
                "duplicates_removed": dropped,
                "unique_files": len(unique),
                "scan_ids": len({scan_id(f.name) for f in unique}),
                **{name: len(split[name]) for name in SPLITS},
            }
            logger.info(f"{class_dir.name}: {report['classes'][class_dir.name]}")

        self.config.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.report_path.write_text(json.dumps(report, indent=4))
        return report
