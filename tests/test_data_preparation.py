from pathlib import Path

from cnnClassifier.components.data_preparation import dedupe, group_split, scan_id


def test_scan_id_strips_slice_and_copy_suffixes():
    assert scan_id("000005 (3).png") == "000005"
    assert scan_id("000024.png") == "000024"
    assert scan_id("10 - Copy (2) - Copy.png") == "10"
    assert scan_id("n8 (2) - Copy.jpg") == "n8"


def test_dedupe_drops_byte_identical_files(tmp_path: Path):
    for name, content in [("a.png", b"x"), ("a - Copy.png", b"x"), ("b.png", b"y")]:
        (tmp_path / name).write_bytes(content)
    unique, dropped = dedupe(list(tmp_path.iterdir()))
    assert dropped == 1
    assert len(unique) == 2


def test_group_split_keeps_each_scan_in_one_split(tmp_path: Path):
    files = []
    for scan in range(20):
        for slice_ in range(5):
            f = tmp_path / f"{scan:06d} ({slice_}).png"
            f.write_bytes(f"{scan}-{slice_}".encode())
            files.append(f)

    split = group_split(files, val=0.15, test=0.15, seed=42)

    ids = {name: {scan_id(f.name) for f in fs} for name, fs in split.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
    assert sum(len(fs) for fs in split.values()) == len(files)
    assert split["test"] and split["val"]


def test_group_split_is_reproducible(tmp_path: Path):
    files = []
    for scan in range(10):
        f = tmp_path / f"{scan}.png"
        f.write_bytes(str(scan).encode())
        files.append(f)
    assert group_split(files, 0.2, 0.2, seed=1) == group_split(files, 0.2, 0.2, seed=1)
