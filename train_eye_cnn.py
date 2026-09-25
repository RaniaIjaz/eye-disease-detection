"""Train and evaluate an enhanced custom CNN for four eye-disease classes.

The pipeline intentionally does not modify the original notebook or dataset ZIP.
It extracts into a cache directory, removes exact duplicate images, keeps paired
left/right images from the same patient in one split, trains a custom residual
CNN, and writes all models, metrics, manifests, and figures to a new run folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers, regularizers


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


@dataclass(frozen=True)
class RunConfig:
    data_zip: str
    output_dir: str
    cache_dir: str
    image_size: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    dropout: float
    seed: int
    smoke_test: bool


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a leakage-resistant custom eye-disease CNN."
    )
    parser.add_argument("--data-zip", type=Path, default=Path("archive (3).zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/enhanced_cnn"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/eye_dataset"))
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one epoch on a small class-balanced subset and generate every artifact.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path.expanduser().resolve()


def unique_output_dir(requested: Path) -> Path:
    requested = resolve_path(requested)
    if not requested.exists() or not any(requested.iterdir()):
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    alternate = requested.with_name(f"{requested.name}-{suffix}")
    alternate.mkdir(parents=True, exist_ok=False)
    return alternate


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(zip_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extraction_complete.json"
    signature = {
        "zip_name": zip_path.name,
        "zip_size_bytes": zip_path.stat().st_size,
        "zip_mtime_ns": zip_path.stat().st_mtime_ns,
    }
    if marker.exists():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == signature:
                return destination
        except (json.JSONDecodeError, OSError):
            pass

    destination_root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise ValueError(f"Unsafe ZIP entry: {member.filename}")
        archive.extractall(destination)

    marker.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    return destination


def find_dataset_root(extraction_dir: Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    directories = [extraction_dir, *[p for p in extraction_dir.rglob("*") if p.is_dir()]]
    for directory in directories:
        class_dirs = [child for child in directory.iterdir() if child.is_dir()]
        class_image_count = 0
        valid_class_count = 0
        for class_dir in class_dirs:
            count = sum(
                1
                for item in class_dir.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            )
            if count:
                valid_class_count += 1
                class_image_count += count
        if valid_class_count >= 2:
            candidates.append((class_image_count, directory))
    if not candidates:
        raise FileNotFoundError("Could not find class directories inside the extracted ZIP.")
    return max(candidates, key=lambda item: item[0])[1]


def patient_key(path: Path) -> str:
    match = re.match(r"^(.+?)_(left|right)$", path.stem, flags=re.IGNORECASE)
    base = match.group(1) if match else path.stem
    return base.strip().lower()


def verify_image(path: Path) -> tuple[bool, str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except Exception as exc:  # Pillow raises several format-specific exceptions.
        return False, f"{type(exc).__name__}: {exc}"


def build_manifest(dataset_root: Path, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    corrupt_rows: list[dict[str, str]] = []
    class_dirs = sorted(
        [directory for directory in dataset_root.iterdir() if directory.is_dir()],
        key=lambda item: item.name.lower(),
    )
    for class_dir in class_dirs:
        image_paths = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ],
            key=lambda item: item.name.lower(),
        )
        for path in image_paths:
            valid, error = verify_image(path)
            if not valid:
                corrupt_rows.append({"path": str(path), "error": error or "unknown"})
                continue
            rows.append(
                {
                    "path": str(path.resolve()),
                    "filename": path.name,
                    "class_name": class_dir.name,
                    "patient_key": patient_key(path),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise ValueError("No valid images were found in the dataset.")

    if corrupt_rows:
        pd.DataFrame(corrupt_rows).to_csv(output_dir / "corrupt_images.csv", index=False)

    duplicate_rows: list[pd.DataFrame] = []
    conflicting_rows: list[pd.DataFrame] = []
    keep_indices: list[int] = []
    for _, group in manifest.groupby("sha256", sort=True):
        group = group.sort_values("path")
        if len(group) == 1:
            keep_indices.append(int(group.index[0]))
        elif group["class_name"].nunique() > 1:
            conflicting_rows.append(group)
        else:
            keep_indices.append(int(group.index[0]))
            duplicate_rows.append(group.iloc[1:])

    if duplicate_rows:
        pd.concat(duplicate_rows, ignore_index=True).to_csv(
            output_dir / "exact_duplicates_removed.csv", index=False
        )
    if conflicting_rows:
        pd.concat(conflicting_rows, ignore_index=True).to_csv(
            output_dir / "conflicting_duplicate_labels_excluded.csv", index=False
        )

    manifest = manifest.loc[sorted(keep_indices)].reset_index(drop=True)
    class_names = sorted(manifest["class_name"].unique(), key=str.lower)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    manifest["class_index"] = manifest["class_name"].map(class_to_index).astype(int)
    return manifest


def merge_duplicate_patient_groups(manifest: pd.DataFrame) -> pd.DataFrame:
    patient_values = sorted(manifest["patient_key"].unique())
    union_find = UnionFind(patient_values)
    for _, group in manifest.groupby("sha256"):
        patients = sorted(group["patient_key"].unique())
        for patient in patients[1:]:
            union_find.union(patients[0], patient)
    result = manifest.copy()
    result["group_id"] = result["patient_key"].map(union_find.find)
    return result


def stable_tiebreaker(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def assign_grouped_stratified_splits(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    class_names = sorted(manifest["class_name"].unique(), key=str.lower)
    split_names = list(SPLIT_FRACTIONS)
    total_counts = (
        manifest["class_name"].value_counts().reindex(class_names, fill_value=0).to_numpy(float)
    )
    targets = np.stack(
        [total_counts * SPLIT_FRACTIONS[split_name] for split_name in split_names]
    )
    current = np.zeros_like(targets)

    groups: list[tuple[str, np.ndarray]] = []
    for group_id, group in manifest.groupby("group_id"):
        counts = (
            group["class_name"].value_counts().reindex(class_names, fill_value=0).to_numpy(float)
        )
        groups.append((str(group_id), counts))

    random.Random(seed).shuffle(groups)
    groups.sort(
        key=lambda item: (
            -int(item[1].sum()),
            -int(np.count_nonzero(item[1])),
            stable_tiebreaker(seed, item[0]),
        )
    )

    assignments: dict[str, str] = {}
    for group_id, group_counts in groups:
        scored_candidates: list[tuple[float, str, int]] = []
        for split_index, split_name in enumerate(split_names):
            candidate = current.copy()
            candidate[split_index] += group_counts
            normalized_error = np.mean(
                np.square((candidate - targets) / np.maximum(targets, 1.0))
            )
            overshoot = np.maximum(candidate - targets, 0.0) / np.maximum(targets, 1.0)
            score = float(normalized_error + (4.0 * np.mean(np.square(overshoot))))
            scored_candidates.append((score, split_name, split_index))
        _, chosen_name, chosen_index = min(scored_candidates, key=lambda item: (item[0], item[1]))
        current[chosen_index] += group_counts
        assignments[group_id] = chosen_name

    result = manifest.copy()
    result["split"] = result["group_id"].map(assignments)

    leakage = result.groupby("group_id")["split"].nunique()
    if int(leakage.max()) != 1:
        raise AssertionError("Patient-group leakage detected across dataset splits.")
    duplicate_leakage = result.groupby("sha256")["split"].nunique()
    if int(duplicate_leakage.max()) != 1:
        raise AssertionError("Exact-duplicate leakage detected across dataset splits.")
    coverage = result.groupby(["split", "class_name"]).size().unstack(fill_value=0)
    if (coverage == 0).any().any():
        raise ValueError(f"At least one split is missing a class:\n{coverage}")
    return result


def balanced_smoke_subset(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    per_class = {"train": 4, "validation": 2, "test": 2}
    pieces: list[pd.DataFrame] = []
    for split_name, split_frame in manifest.groupby("split"):
        take = per_class[str(split_name)]
        for _, class_frame in split_frame.groupby("class_name"):
            pieces.append(
                class_frame.sample(
                    n=min(take, len(class_frame)), random_state=seed
                )
            )
    return pd.concat(pieces, ignore_index=True).sort_values(
        ["split", "class_name", "path"]
    )


def write_split_artifacts(manifest: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    manifest.to_csv(output_dir / "dataset_manifest.csv", index=False)
    summary = (
        manifest.groupby(["split", "class_name"], observed=True)
        .size()
        .rename("image_count")
        .reset_index()
    )
    summary.to_csv(output_dir / "split_summary.csv", index=False)
    return summary


def configure_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def decode_and_resize(path: tf.Tensor, label: tf.Tensor, image_size: int):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, image_size, image_size, antialias=True)
    return image, label


def make_dataset(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    seed: int,
    training: bool,
) -> tf.data.Dataset:
    ordered = frame.sort_values("path")
    paths = ordered["path"].astype(str).to_numpy()
    labels = ordered["class_index"].astype(np.int32).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(ordered), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        lambda path, label: decode_and_resize(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)


def conv_bn_relu(
    inputs: tf.Tensor,
    filters: int,
    kernel_size: int,
    stride: int,
    weight_decay: float,
    name: str,
) -> tf.Tensor:
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=stride,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(weight_decay),
        name=f"{name}_conv",
    )(inputs)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("relu", name=f"{name}_relu")(x)


def squeeze_excite(inputs: tf.Tensor, filters: int, name: str) -> tf.Tensor:
    reduced_filters = max(filters // 8, 8)
    scale = layers.GlobalAveragePooling2D(name=f"{name}_gap")(inputs)
    scale = layers.Reshape((1, 1, filters), name=f"{name}_reshape")(scale)
    scale = layers.Dense(reduced_filters, activation="relu", name=f"{name}_reduce")(scale)
    scale = layers.Dense(filters, activation="sigmoid", name=f"{name}_expand")(scale)
    return layers.Multiply(name=f"{name}_scale")([inputs, scale])


def residual_se_block(
    inputs: tf.Tensor,
    filters: int,
    stride: int,
    weight_decay: float,
    name: str,
) -> tf.Tensor:
    shortcut = inputs
    x = conv_bn_relu(inputs, filters, 3, stride, weight_decay, f"{name}_a")
    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=regularizers.l2(weight_decay),
        name=f"{name}_b_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_b_bn")(x)
    x = squeeze_excite(x, filters, f"{name}_se")

    input_channels = int(inputs.shape[-1])
    if stride != 1 or input_channels != filters:
        shortcut = layers.Conv2D(
            filters,
            1,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay),
            name=f"{name}_shortcut_conv",
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)

    x = layers.Add(name=f"{name}_add")([x, shortcut])
    return layers.Activation("relu", name=f"{name}_out")(x)


def build_model(
    image_size: int,
    num_classes: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    seed: int,
) -> keras.Model:
    inputs = keras.Input((image_size, image_size, 3), name="image")
    augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.03, fill_mode="reflect", seed=seed + 1),
            layers.RandomZoom(0.10, fill_mode="reflect", seed=seed + 2),
            layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed + 3),
            layers.RandomContrast(0.10, seed=seed + 4),
        ],
        name="training_augmentation",
    )
    x = augmentation(inputs)
    x = conv_bn_relu(x, 32, 5, 2, weight_decay, "stem")
    x = residual_se_block(x, 32, 1, weight_decay, "block1")
    x = residual_se_block(x, 64, 2, weight_decay, "block2")
    x = residual_se_block(x, 128, 2, weight_decay, "block3")
    x = residual_se_block(x, 192, 2, weight_decay, "block4")
    x = layers.Activation("linear", name="last_conv_features")(x)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(dropout, seed=seed + 5, name="feature_dropout")(x)
    x = layers.Dense(
        128,
        use_bias=False,
        kernel_regularizer=regularizers.l2(weight_decay),
        name="classifier_dense",
    )(x)
    x = layers.BatchNormalization(name="classifier_bn")(x)
    x = layers.Activation("relu", name="classifier_relu")(x)
    x = layers.Dropout(dropout * 0.75, seed=seed + 6, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="enhanced_eye_disease_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def save_model_summary(model: keras.Model, output_path: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append, expand_nested=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def class_weights_from_frame(frame: pd.DataFrame) -> dict[int, float]:
    labels = frame["class_index"].to_numpy()
    classes = np.sort(np.unique(labels))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def plot_history(history: dict[str, list[float]], output_dir: Path) -> None:
    epoch_values = np.arange(1, len(history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epoch_values, history["accuracy"], marker="o", label="Training")
    axes[0].plot(epoch_values, history["val_accuracy"], marker="o", label="Validation")
    axes[0].set(title="Training and validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(epoch_values, history["loss"], marker="o", label="Training")
    axes[1].plot(epoch_values, history["val_loss"], marker="o", label="Validation")
    axes[1].set(title="Training and validation loss", xlabel="Epoch", ylabel="Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "training_history.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_class_distribution(summary: pd.DataFrame, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=summary, x="class_name", y="image_count", hue="split", ax=axis)
    axis.set(title="Class distribution by split", xlabel="Class", ylabel="Images")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output_dir / "class_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def collect_predictions(
    model: keras.Model, dataset: tf.data.Dataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    labels: list[np.ndarray] = []
    for _, batch_labels in dataset:
        labels.append(batch_labels.numpy())
    y_true = np.concatenate(labels).astype(int)

    first_batch = next(iter(dataset.take(1)))[0]
    _ = model(first_batch, training=False)
    start = time.perf_counter()
    probabilities = model.predict(dataset, verbose=0)
    elapsed_seconds = time.perf_counter() - start
    y_pred = probabilities.argmax(axis=1).astype(int)
    return y_true, y_pred, probabilities, elapsed_seconds


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> None:
    raw = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    normalized = confusion_matrix(
        y_true, y_pred, labels=np.arange(len(class_names)), normalize="true"
    )
    for matrix, filename, formatting, title in [
        (raw, "confusion_matrix.png", "d", "Confusion matrix"),
        (
            normalized,
            "confusion_matrix_normalized.png",
            ".2f",
            "Normalized confusion matrix",
        ),
    ]:
        figure, axis = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=formatting,
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axis,
        )
        axis.set(title=title, xlabel="Predicted", ylabel="True")
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)


def plot_per_class_metrics(
    report: dict[str, object], class_names: list[str], output_dir: Path
) -> None:
    rows = []
    for class_name in class_names:
        values = report[class_name]
        assert isinstance(values, dict)
        for metric in ("precision", "recall", "f1-score"):
            rows.append(
                {"class_name": class_name, "metric": metric, "value": values[metric]}
            )
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=frame, x="class_name", y="value", hue="metric", ax=axis)
    axis.set(
        title="Per-class precision, recall, and F1-score",
        xlabel="Class",
        ylabel="Score",
        ylim=(0.0, 1.0),
    )
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output_dir / "per_class_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confidence_and_correctness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
) -> None:
    confidence = probabilities.max(axis=1)
    correct = y_true == y_pred
    frame = pd.DataFrame(
        {
            "confidence": confidence,
            "result": np.where(correct, "Correct", "Incorrect"),
        }
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.histplot(
        data=frame,
        x="confidence",
        hue="result",
        bins=20,
        multiple="layer",
        stat="count",
        common_norm=False,
        ax=axis,
    )
    axis.set(title="Prediction confidence distribution", xlabel="Maximum probability")
    figure.tight_layout()
    figure.savefig(output_dir / "confidence_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    counts = frame["result"].value_counts().reindex(["Correct", "Incorrect"], fill_value=0)
    figure, axis = plt.subplots(figsize=(7, 5))
    bars = axis.bar(counts.index, counts.values, color=["#2a9d8f", "#e76f51"])
    axis.bar_label(bars)
    axis.set(title="Correct versus incorrect predictions", ylabel="Images")
    figure.tight_layout()
    figure.savefig(output_dir / "correct_vs_incorrect.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_roc_and_precision_recall(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    binary_true = label_binarize(y_true, classes=np.arange(len(class_names)))
    curve_metrics: dict[str, dict[str, float]] = {}

    roc_figure, roc_axis = plt.subplots(figsize=(8, 7))
    pr_figure, pr_axis = plt.subplots(figsize=(8, 7))
    for index, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(binary_true[:, index], probabilities[:, index])
        precision, recall, _ = precision_recall_curve(
            binary_true[:, index], probabilities[:, index]
        )
        roc_auc = float(auc(fpr, tpr))
        pr_auc = float(auc(recall[::-1], precision[::-1]))
        curve_metrics[class_name] = {"roc_auc": roc_auc, "pr_auc": pr_auc}
        roc_axis.plot(fpr, tpr, label=f"{class_name} (AUC={roc_auc:.3f})")
        pr_axis.plot(recall, precision, label=f"{class_name} (AUC={pr_auc:.3f})")

    roc_axis.plot([0, 1], [0, 1], "--", color="gray")
    roc_axis.set(title="One-vs-rest ROC curves", xlabel="False positive rate", ylabel="True positive rate")
    roc_axis.legend(fontsize=8)
    roc_axis.grid(alpha=0.25)
    roc_figure.tight_layout()
    roc_figure.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(roc_figure)

    pr_axis.set(title="One-vs-rest precision-recall curves", xlabel="Recall", ylabel="Precision")
    pr_axis.legend(fontsize=8)
    pr_axis.grid(alpha=0.25)
    pr_figure.tight_layout()
    pr_figure.savefig(output_dir / "precision_recall_curves.png", dpi=300, bbox_inches="tight")
    plt.close(pr_figure)
    return curve_metrics


def load_display_image(path: str, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((image_size, image_size))
        canvas = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))
        left = (image_size - image.width) // 2
        top = (image_size - image.height) // 2
        canvas.paste(image, (left, top))
        return np.asarray(canvas, dtype=np.float32) / 255.0


def plot_misclassified_examples(
    test_frame: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    image_size: int,
    output_dir: Path,
) -> None:
    ordered = test_frame.sort_values("path").reset_index(drop=True)
    incorrect_indices = np.flatnonzero(y_true != y_pred)
    selected = incorrect_indices[:12]
    figure, axes = plt.subplots(3, 4, figsize=(14, 11))
    axes_array = np.asarray(axes).reshape(-1)
    if len(selected) == 0:
        for axis in axes_array:
            axis.axis("off")
        axes_array[0].text(0.5, 0.5, "No misclassified test images", ha="center", va="center")
    else:
        for axis, item_index in zip(axes_array, selected):
            row = ordered.iloc[int(item_index)]
            axis.imshow(load_display_image(str(row["path"]), image_size))
            confidence = probabilities[item_index, y_pred[item_index]]
            axis.set_title(
                f"True: {class_names[y_true[item_index]]}\n"
                f"Pred: {class_names[y_pred[item_index]]} ({confidence:.3f})",
                fontsize=9,
            )
            axis.axis("off")
        for axis in axes_array[len(selected) :]:
            axis.axis("off")
    figure.suptitle("Misclassified test examples")
    figure.tight_layout()
    figure.savefig(output_dir / "misclassified_examples.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def make_gradcam_heatmap(
    model: keras.Model,
    image_batch: np.ndarray,
    predicted_index: int,
) -> np.ndarray:
    grad_model = keras.Model(
        model.inputs,
        [model.get_layer("last_conv_features").output, model.output],
    )
    image_tensor = tf.convert_to_tensor(image_batch)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor, training=False)
        class_score = predictions[:, predicted_index]
    gradients = tape.gradient(class_score, conv_outputs)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_gradients, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
    return heatmap.numpy()


def save_gradcam_examples(
    model: keras.Model,
    test_frame: pd.DataFrame,
    y_pred: np.ndarray,
    class_names: list[str],
    image_size: int,
    output_dir: Path,
) -> None:
    ordered = test_frame.sort_values("path").reset_index(drop=True)
    selected_indices: list[int] = []
    for class_index in range(len(class_names)):
        matching = np.flatnonzero(ordered["class_index"].to_numpy() == class_index)
        if len(matching):
            selected_indices.append(int(matching[0]))

    figure, axes = plt.subplots(2, len(selected_indices), figsize=(4 * len(selected_indices), 8))
    axes_array = np.asarray(axes)
    if axes_array.ndim == 1:
        axes_array = axes_array.reshape(2, -1)
    colormap = plt.get_cmap("jet")

    for column, item_index in enumerate(selected_indices):
        row = ordered.iloc[item_index]
        image = load_display_image(str(row["path"]), image_size)
        predicted_index = int(y_pred[item_index])
        heatmap = make_gradcam_heatmap(model, image[None, ...], predicted_index)
        heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize(
            (image_size, image_size), Image.Resampling.BILINEAR
        )
        colored = colormap(np.asarray(heatmap_image) / 255.0)[..., :3]
        overlay = np.clip((0.55 * image) + (0.45 * colored), 0.0, 1.0)

        axes_array[0, column].imshow(image)
        axes_array[0, column].set_title(f"True: {row['class_name']}")
        axes_array[0, column].axis("off")
        axes_array[1, column].imshow(overlay)
        axes_array[1, column].set_title(f"Grad-CAM: {class_names[predicted_index]}")
        axes_array[1, column].axis("off")

    figure.suptitle("Grad-CAM examples")
    figure.tight_layout()
    figure.savefig(output_dir / "gradcam_examples.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_evaluation_report(
    output_dir: Path,
    model_path: Path,
    class_names: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    inference_seconds: float,
    report: dict[str, object],
    curve_metrics: dict[str, dict[str, float]],
) -> dict[str, object]:
    confidence = probabilities.max(axis=1)
    metrics = {
        "test_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "mean_confidence": float(np.mean(confidence)),
        "correct_predictions": int(np.sum(y_true == y_pred)),
        "incorrect_predictions": int(np.sum(y_true != y_pred)),
        "inference_seconds_total": float(inference_seconds),
        "inference_milliseconds_per_image": float((inference_seconds / len(y_true)) * 1000.0),
        "model_size_bytes": int(model_path.stat().st_size),
        "model_size_megabytes": float(model_path.stat().st_size / (1024 * 1024)),
        "per_class_curves": curve_metrics,
        "classification_report": report,
    }
    (output_dir / "evaluation_metrics.json").write_text(
        json.dumps(json_ready(metrics), indent=2), encoding="utf-8"
    )

    rows = []
    for class_name in class_names:
        class_values = report[class_name]
        rows.append(
            {
                "class_name": class_name,
                "precision": class_values["precision"],
                "recall": class_values["recall"],
                "f1_score": class_values["f1-score"],
                "support": class_values["support"],
                **curve_metrics[class_name],
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "per_class_metrics.csv", index=False)

    report_lines = [
        "# Eye Disease CNN Evaluation",
        "",
        "These values were computed from the saved best model on the held-out test split.",
        "",
        f"- Test images: {metrics['test_images']}",
        f"- Accuracy: {metrics['accuracy']:.6f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.6f}",
        f"- Macro F1: {metrics['macro_f1']:.6f}",
        f"- Correct predictions: {metrics['correct_predictions']}",
        f"- Incorrect predictions: {metrics['incorrect_predictions']}",
        f"- Total measured inference time: {metrics['inference_seconds_total']:.6f} seconds",
        f"- Measured inference time per image: {metrics['inference_milliseconds_per_image']:.6f} ms",
        f"- Saved model size: {metrics['model_size_bytes']} bytes",
        "",
        "This is an academic prototype and is not intended for clinical diagnosis.",
    ]
    (output_dir / "EVALUATION.md").write_text("\n".join(report_lines), encoding="utf-8")
    return metrics


def main() -> int:
    args = parse_args()
    data_zip = resolve_path(args.data_zip)
    if not data_zip.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {data_zip}")

    output_dir = unique_output_dir(args.output_dir)
    cache_dir = resolve_path(args.cache_dir)
    image_size = 96 if args.smoke_test else args.image_size
    batch_size = min(args.batch_size, 4) if args.smoke_test else args.batch_size
    epochs = 1 if args.smoke_test else args.epochs
    configure_reproducibility(args.seed)

    config = RunConfig(
        data_zip=str(data_zip),
        output_dir=str(output_dir),
        cache_dir=str(cache_dir),
        image_size=image_size,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed,
        smoke_test=args.smoke_test,
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    print(f"Run directory: {output_dir}", flush=True)
    print(f"TensorFlow version: {tf.__version__}", flush=True)
    print(f"Devices: {[device.name for device in tf.config.list_physical_devices()]}", flush=True)

    extraction_dir = safe_extract(data_zip, cache_dir / "extracted")
    dataset_root = find_dataset_root(extraction_dir)
    print(f"Dataset root: {dataset_root}", flush=True)

    manifest = build_manifest(dataset_root, output_dir)
    manifest = merge_duplicate_patient_groups(manifest)
    manifest = assign_grouped_stratified_splits(manifest, args.seed)
    full_manifest = manifest.copy()
    if args.smoke_test:
        manifest = balanced_smoke_subset(manifest, args.seed)

    summary = write_split_artifacts(manifest, output_dir)
    plot_class_distribution(summary, output_dir)
    print("Split summary:", flush=True)
    print(summary.to_string(index=False), flush=True)

    class_names = sorted(full_manifest["class_name"].unique(), key=str.lower)
    class_mapping = {str(index): name for index, name in enumerate(class_names)}
    (output_dir / "class_names.json").write_text(
        json.dumps(class_mapping, indent=2), encoding="utf-8"
    )

    train_frame = manifest[manifest["split"] == "train"].copy()
    validation_frame = manifest[manifest["split"] == "validation"].copy()
    test_frame = manifest[manifest["split"] == "test"].copy()

    train_dataset = make_dataset(
        train_frame, image_size, batch_size, args.seed, training=True
    )
    validation_dataset = make_dataset(
        validation_frame, image_size, batch_size, args.seed, training=False
    )
    test_dataset = make_dataset(
        test_frame, image_size, batch_size, args.seed, training=False
    )

    model = build_model(
        image_size=image_size,
        num_classes=len(class_names),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        seed=args.seed,
    )
    save_model_summary(model, output_dir / "model_summary.txt")
    weights = class_weights_from_frame(train_frame)
    (output_dir / "class_weights.json").write_text(
        json.dumps(weights, indent=2), encoding="utf-8"
    )

    best_model_path = output_dir / "best_eye_disease_cnn.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(output_dir / "training_history.csv"),
        keras.callbacks.TerminateOnNaN(),
    ]

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        class_weight=weights,
        callbacks=callbacks,
        verbose=2,
    )
    history_data = {key: [float(value) for value in values] for key, values in history.history.items()}
    (output_dir / "training_history.json").write_text(
        json.dumps(history_data, indent=2), encoding="utf-8"
    )
    plot_history(history_data, output_dir)

    best_model = keras.models.load_model(best_model_path)
    y_true, y_pred, probabilities, inference_seconds = collect_predictions(
        best_model, test_dataset
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    plot_confusion_matrices(y_true, y_pred, class_names, output_dir)
    plot_per_class_metrics(report, class_names, output_dir)
    plot_confidence_and_correctness(y_true, y_pred, probabilities, output_dir)
    curve_metrics = plot_roc_and_precision_recall(
        y_true, probabilities, class_names, output_dir
    )
    plot_misclassified_examples(
        test_frame,
        y_true,
        y_pred,
        probabilities,
        class_names,
        image_size,
        output_dir,
    )
    save_gradcam_examples(
        best_model, test_frame, y_pred, class_names, image_size, output_dir
    )
    metrics = write_evaluation_report(
        output_dir,
        best_model_path,
        class_names,
        y_true,
        y_pred,
        probabilities,
        inference_seconds,
        report,
        curve_metrics,
    )

    run_metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "devices": [device.name for device in tf.config.list_physical_devices()],
        "model_parameters": int(best_model.count_params()),
        "source_zip_sha256": sha256_file(data_zip),
        "valid_images_before_smoke_subset": int(len(full_manifest)),
        "training_images_used": int(len(train_frame)),
        "validation_images_used": int(len(validation_frame)),
        "test_images_used": int(len(test_frame)),
        "metrics": metrics,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(json_ready(run_metadata), indent=2), encoding="utf-8"
    )

    print("Computed evaluation metrics:", flush=True)
    print(json.dumps(json_ready(metrics), indent=2), flush=True)
    print(f"Completed run: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
