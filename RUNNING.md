# Running the enhanced eye-disease CNN

The original `Untitled2.ipynb` and `archive (3).zip` are not modified.

## Environment setup

```powershell
py -3.13 -m venv D:\eyediseasedetection\.venv
D:\eyediseasedetection\.venv\Scripts\python.exe -m pip install -r D:\eyediseasedetection\requirements.txt
```

## Smoke test

```powershell
D:\eyediseasedetection\.venv\Scripts\python.exe D:\eyediseasedetection\train_eye_cnn.py --data-zip "D:\eyediseasedetection\archive (3).zip" --output-dir "D:\eyediseasedetection\runs\smoke_test" --smoke-test
```

## Full training

```powershell
D:\eyediseasedetection\.venv\Scripts\python.exe D:\eyediseasedetection\train_eye_cnn.py --data-zip "D:\eyediseasedetection\archive (3).zip" --output-dir "D:\eyediseasedetection\runs\enhanced_cnn"
```

If the requested output folder already contains files, the pipeline creates a
timestamped sibling folder rather than overwriting a previous run.

## Generated artifacts

Each run writes the best Keras model, class mapping, dataset manifest, split
summary, training history, confusion matrices, per-class metrics, ROC and
precision-recall curves, confidence and correctness charts, misclassified
examples, Grad-CAM examples, measured inference timing, model size, and JSON
run metadata.

This is an academic prototype and is not intended for clinical diagnosis.
