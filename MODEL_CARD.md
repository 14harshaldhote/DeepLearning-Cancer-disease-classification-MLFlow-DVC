# Model card: Chest CT adenocarcinoma classifier

## Summary

| | |
|---|---|
| Task | Binary image classification of a single chest CT slice: adenocarcinoma vs normal |
| Model | VGG16 pretrained on ImageNet (convolutional layers frozen) + global average pooling, dropout 0.3, softmax head |
| Input | One RGB image, resized to 224 x 224, pixel values scaled to [0, 1] |
| Output | Class probabilities, a "needs review" flag and a Grad-CAM heatmap |
| Framework | TensorFlow / Keras 2.15, trained on CPU |
| Version | 2.0.0 (see `dvc.lock` for the exact data, code and parameter hashes) |

## Intended use

- A portfolio and learning project that demonstrates an end-to-end MLOps workflow:
  versioned data pipeline (DVC), experiment tracking (MLflow), a quality gate before
  deployment, an explainable API (FastAPI + Grad-CAM), CI/CD and containerisation.
- **Not for clinical use.** It is not a medical device, has not been validated on any
  clinical population, and must not inform a diagnosis or treatment decision.

## Data

Public chest CT slice dataset (bundled as `research/Chest-CT-Scan-data.zip`, 343 files).

| Class | Raw files | Exact duplicates removed | Unique images | Scan IDs | Train | Val | Test |
|---|---|---|---|---|---|---|---|
{{DATA_ROWS}}

Two problems in the raw data would have inflated any score measured on a random split:

1. **Duplicates.** 93 of the 148 "normal" files are byte-identical copies
   (`10 - Copy.png`, `10 - Copy (2).png`, ...). They are removed by content hash.
2. **Several slices per scan.** Files such as `000005 (3).png` and `000005 (9).png`
   come from the same scan. The split is made by scan ID (the file name stem), so no
   scan appears in more than one of train / val / test.

The previous version of this project reported 100% accuracy. It evaluated on a
validation split that overlapped the training images and contained duplicates, so that
number was not a real held-out result.

## Training

- Augmentation: rotation, shifts, shear, zoom, horizontal flip.
- Class weights inversely proportional to class frequency (normal is the minority class
  after de-duplication).
- Adam, learning rate {{LR}}, batch size {{BATCH}}, up to {{EPOCHS}} epochs with early
  stopping on validation loss (patience {{PATIENCE}}); best weights restored.
- Seed {{SEED}} for the split and for training.

## Evaluation (held-out test set, {{N_TEST}} images)

| Metric | Value |
|---|---|
| Accuracy | {{ACCURACY}} |
| Sensitivity (cancer recall) | {{SENSITIVITY}} |
| Specificity (normal recall) | {{SPECIFICITY}} |
| ROC-AUC | {{AUC}} |
| Macro F1 | {{F1}} |

Confusion matrix (rows = actual, columns = predicted): {{CM}}

The test set is small, so each misclassified image moves accuracy by about
{{STEP}} percentage points. Treat these numbers as indicative, not as a performance guarantee.

## Human oversight

- Predictions with confidence below {{THRESHOLD}} are flagged **Needs expert review** in the
  API and the dashboard.
- Every prediction returns a Grad-CAM heatmap so a reviewer can see which regions drove it.
- The API keeps an in-memory audit trail of recent predictions (time, label, confidence,
  latency); images are not stored.

## Known limitations and risks

- **Small, single-source dataset.** A few hundred unique slices from one public source. The
  model will not generalise to other scanners, protocols or populations without retraining
  and proper validation.
- **Shortcut learning.** The two classes differ in framing, contrast and field of view, not
  only in pathology. Grad-CAM sometimes highlights regions outside the lungs, which suggests
  the model may partly rely on such cues.
- **Slice-level only.** Real CT reading uses the whole 3D volume, prior studies and clinical
  context.
- **Two classes only.** Other cancer types (squamous cell, large cell) and other findings are
  out of scope; the model will still force them into one of the two classes.

## Regulatory note

Software that informs clinical decisions is regulated as a medical device (for example
under the EU MDR, UK MDR and FDA rules), and AI used in that role counts as high-risk under
the EU AI Act. This project is a research demo and does not meet those requirements. The
model card, human-review flag, explanations and audit trail illustrate the kind of
transparency and oversight those rules expect.
