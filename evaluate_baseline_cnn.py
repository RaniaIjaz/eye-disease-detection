"""Reproduce the original notebook CNN on the enhanced run's locked split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow import keras
from tensorflow.keras import layers


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def decode_like_original(path: tf.Tensor, label: tf.Tensor, image_size: int):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size], method="nearest")
    image = tf.cast(image, tf.float32) / 255.0
    return image, tf.one_hot(label, depth=4)


def make_dataset(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    seed: int,
    training: bool,
    repeat: bool,
) -> tf.data.Dataset:
    ordered = frame.sort_values("path")
    paths = ordered["path"].astype(str).to_numpy()
    labels = ordered["class_index"].astype(np.int32).to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(ordered), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.map(
        lambda path, label: decode_like_original(path, label, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=not training,
    )
    dataset = dataset.batch(batch_size, drop_remainder=False)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_original_model(image_size: int, num_classes: int) -> keras.Model:
    return keras.Sequential(
        [
            keras.Input((image_size, image_size, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="original_notebook_baseline_cnn",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("runs/enhanced_cnn/dataset_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baseline_same_split"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--validation-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_reproducibility(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    class_names = (
        manifest[["class_index", "class_name"]]
        .drop_duplicates()
        .sort_values("class_index")["class_name"]
        .tolist()
    )
    train_frame = manifest[manifest["split"] == "train"].copy()
    validation_frame = manifest[manifest["split"] == "validation"].copy()
    test_frame = manifest[manifest["split"] == "test"].copy().sort_values("path")

    train_dataset = make_dataset(
        train_frame, args.image_size, args.batch_size, args.seed, training=True, repeat=True
    )
    validation_dataset = make_dataset(
        validation_frame, args.image_size, args.batch_size, args.seed, training=False, repeat=True
    )
    test_dataset = make_dataset(
        test_frame, args.image_size, args.batch_size, args.seed, training=False, repeat=False
    )

    model = build_original_model(args.image_size, len(class_names))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_dataset,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        validation_data=validation_dataset,
        validation_steps=args.validation_steps,
        verbose=2,
    )

    probabilities = model.predict(test_dataset, verbose=1)
    predictions = probabilities.argmax(axis=1)
    true_labels = test_frame["class_index"].astype(int).to_numpy()
    report = classification_report(
        true_labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    result = {
        "description": "Reproduced original notebook architecture and training settings on the enhanced run's locked split; these are not recovered historical weights.",
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "train_images": int(len(train_frame)),
        "validation_images": int(len(validation_frame)),
        "test_images": int(len(test_frame)),
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch,
        "validation_steps": args.validation_steps,
        "model_parameters": int(model.count_params()),
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "macro_f1": float(f1_score(true_labels, predictions, average="macro", zero_division=0)),
        "classification_report": report,
        "training_history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    model.save(args.output_dir / "baseline_cnn_same_split.keras")
    (args.output_dir / "baseline_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(history.history).to_csv(args.output_dir / "training_history.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
