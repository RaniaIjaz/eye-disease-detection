"""Compute filename-grouping and exact-duplicate statistics for the eye dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PATIENT_PATTERN = re.compile(r"^(?P<patient_id>.+)_(?P<side>left|right)$", re.IGNORECASE)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(".cache/eye_dataset/extracted/dataset"),
    )
    parser.add_argument(
        "--retained-manifest",
        type=Path,
        default=Path("runs/enhanced_cnn/dataset_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/results/data_quality_audit.json"),
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    corrupt: list[dict[str, str]] = []
    for class_dir in sorted(p for p in args.dataset_root.iterdir() if p.is_dir()):
        for path in sorted(
            p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ):
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:
                corrupt.append({"path": str(path.resolve()), "error": f"{type(exc).__name__}: {exc}"})
                continue
            match = PATIENT_PATTERN.match(path.stem)
            rows.append(
                {
                    "path": str(path.resolve()),
                    "filename": path.name,
                    "class_name": class_dir.name,
                    "sha256": sha256_file(path),
                    "patient_pattern_matched": match is not None,
                    "patient_id": match.group("patient_id").strip().lower() if match else None,
                    "side": match.group("side").lower() if match else None,
                    "extension": path.suffix.lower(),
                }
            )

    raw = pd.DataFrame(rows)
    retained = pd.read_csv(args.retained_manifest)
    retained_paths = set(retained["path"].astype(str))
    raw["retained"] = raw["path"].isin(retained_paths)
    retained_raw = raw[raw["retained"]].copy()

    duplicate_groups = [group for _, group in raw.groupby("sha256") if len(group) > 1]
    conflicting_groups = [group for group in duplicate_groups if group["class_name"].nunique() > 1]
    same_label_groups = [group for group in duplicate_groups if group["class_name"].nunique() == 1]

    grouped = retained_raw[retained_raw["patient_pattern_matched"]]
    individual = retained_raw[~retained_raw["patient_pattern_matched"]]
    total_retained = len(retained_raw)
    grouped_count = len(grouped)

    patient_sizes = grouped.groupby("patient_id").size() if not grouped.empty else pd.Series(dtype=int)
    paired_patient_ids = int((patient_sizes >= 2).sum())
    singleton_patient_ids = int((patient_sizes == 1).sum())

    unmatched_stem_counts = individual.assign(
        stem=individual["filename"].map(lambda value: Path(value).stem.lower())
    ).groupby("stem").size()

    conflict_records = []
    for group in conflicting_groups:
        conflict_records.append(
            {
                "sha256": str(group["sha256"].iloc[0]),
                "files": group[["filename", "class_name", "path"]].to_dict(orient="records"),
            }
        )

    result = {
        "dataset_root": str(args.dataset_root.resolve()),
        "valid_source_images": int(len(raw)),
        "corrupt_images": int(len(corrupt)),
        "retained_images_after_duplicate_handling": int(total_retained),
        "patient_filename_regex": PATIENT_PATTERN.pattern,
        "patient_pattern_matched_images": int(grouped_count),
        "patient_pattern_matched_percentage": float(100.0 * grouped_count / total_retained),
        "individual_images_without_patient_pattern": int(len(individual)),
        "individual_images_without_patient_pattern_percentage": float(100.0 * len(individual) / total_retained),
        "matched_unique_patient_ids": int(grouped["patient_id"].nunique()),
        "matched_patient_ids_with_two_or_more_images": paired_patient_ids,
        "matched_patient_ids_with_one_image": singleton_patient_ids,
        "matched_images_by_class": {
            str(key): int(value) for key, value in grouped["class_name"].value_counts().sort_index().items()
        },
        "matched_images_by_extension": {
            str(key): int(value) for key, value in grouped["extension"].value_counts().sort_index().items()
        },
        "matched_filename_examples": grouped["filename"].sort_values().head(12).tolist(),
        "unmatched_filename_examples": individual["filename"].sort_values().head(12).tolist(),
        "unmatched_stems_repeated": {
            str(key): int(value) for key, value in unmatched_stem_counts[unmatched_stem_counts > 1].items()
        },
        "exact_duplicate_content_groups": int(len(duplicate_groups)),
        "image_files_in_exact_duplicate_groups": int(sum(len(group) for group in duplicate_groups)),
        "same_label_duplicate_content_groups": int(len(same_label_groups)),
        "same_label_duplicate_files_removed": int(sum(len(group) - 1 for group in same_label_groups)),
        "conflicting_label_duplicate_content_groups": int(len(conflicting_groups)),
        "conflicting_label_image_files_excluded": int(sum(len(group) for group in conflicting_groups)),
        "conflicting_label_records": conflict_records,
        "corrupt_records": corrupt,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
