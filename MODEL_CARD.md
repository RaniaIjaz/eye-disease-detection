# Enhanced Eye Disease CNN — Model Card

## Overview

This academic image-classification model distinguishes four retinal-image
classes: cataract, diabetic retinopathy, glaucoma, and normal. It is a custom
CNN rather than a pretrained transfer-learning model. The architecture uses
residual blocks, squeeze-and-excitation channel attention, batch normalization,
global average pooling, L2 regularization, and dropout.

The model is an academic prototype. It is not intended or validated for
clinical diagnosis.

## Dataset processing

- Images found in the source archive: 4,217
- Images retained after validation: 4,213
- Images excluded because identical content had conflicting labels: 4
- Split method: deterministic grouping by filename-derived patient identifier
- Paired files such as `1102_left.jpg` and `1102_right.jpg` remain in one split
- Exact duplicates cannot cross splits

| Split | Cataract | Diabetic retinopathy | Glaucoma | Normal | Total |
|---|---:|---:|---:|---:|---:|
| Train | 726 | 770 | 703 | 752 | 2,951 |
| Validation | 155 | 164 | 151 | 161 | 631 |
| Test | 155 | 164 | 151 | 161 | 631 |

## Training

- Input resolution: 192 × 192 RGB
- Maximum configured epochs: 40
- Epochs completed before early stopping: 28
- Best checkpoint selected by validation loss: epoch 21
- Optimizer: Adam
- Initial learning rate: 0.0003
- Trainable parameters: 931,004

## Held-out test results

All values below were computed from the saved best checkpoint on the 631-image
test split.

| Metric | Computed value |
|---|---:|
| Accuracy | 0.7543581616481775 |
| Balanced accuracy | 0.7548752538125402 |
| Macro precision | 0.7882869583099161 |
| Macro recall | 0.7548752538125402 |
| Macro F1 | 0.7553502811017124 |
| Weighted F1 | 0.7572704308087379 |
| Correct predictions | 476 |
| Incorrect predictions | 155 |

### Per-class results

| Class | Precision | Recall | F1 | ROC-AUC | PR-AUC | Support |
|---|---:|---:|---:|---:|---:|---:|
| Cataract | 0.9078014184397163 | 0.8258064516129032 | 0.8648648648648649 | 0.9787611818921117 | 0.9435993126016564 | 155 |
| Diabetic retinopathy | 0.967948717948718 | 0.9207317073170732 | 0.94375 | 0.9939416096516425 | 0.9859644654594364 | 164 |
| Glaucoma | 0.5150214592274678 | 0.7947019867549668 | 0.625 | 0.8721854304635762 | 0.6787608555882154 | 151 |
| Normal | 0.7623762376237624 | 0.4782608695652174 | 0.5877862595419847 | 0.8827672789744945 | 0.7312243293958922 | 161 |

## Measured artifact and inference information

- Saved model size: 11,437,821 bytes
- Total measured prediction time for 631 test images: 5.806460199994035 seconds
- Measured prediction time per image: 9.20199714737565 milliseconds
- Measurement device reported by TensorFlow: `/physical_device:CPU:0`

## Files

- Model: `models/best_eye_disease_cnn.keras`
- Class mapping: `models/class_names.json`
- Model summary: `models/model_summary.txt`
- Complete computed results: `docs/results/`
- Reproducible pipeline: `train_eye_cnn.py`

## Limitations

- The patient grouping is inferred from the filename convention rather than a
  separately supplied clinical patient table.
- The model has not been externally validated on an independent institution or
  acquisition device.
- Glaucoma precision and normal-class recall are materially lower than the
  corresponding cataract and diabetic-retinopathy scores.
- No demographic subgroup, device-shift, calibration, or prospective clinical
  evaluation was available.
- Predictions must not be used for diagnosis or treatment decisions.
