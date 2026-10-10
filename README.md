# MedScan — browser demo

MedScan is an educational demo for exploring medical-imaging model
behavior. It is **not for diagnosis, treatment decisions, emergency triage, or
delaying care**. Outputs are not clinical advice.

## Run the website in a browser

You need Python, Node.js, and npm installed on the development computer. This
starts a local website; no QR code or Expo Go is needed.

1. In the repository root, start the Python API:

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install '.[api]'
   python -m scanexplain
   ```

   The API listens on port `8000`.

2. In another terminal, start the website:

   ```sh
   cd mobile
   npm install
   npx expo start --web
   ```

3. Open the local browser URL Expo prints (usually
   `http://localhost:8081`). Use the browser URL only; no QR scan is required.

The repo targets Expo SDK 57. **Demo Cases** is the first tab. Search by one of
six fictional names; only the selected case's details appear, and other names
appear as search matches only while choosing a case. Each case has a generated
demo ID and simulated date/time. These values are interface-only: the public
benchmark has no patient identities or acquisition timestamps, and the image
does not belong to the fictional person. Uploads never inherit that metadata.

The API trains a small 16×16 logistic-regression baseline on the
PneumoniaMNIST training split the first time analysis is requested; validation
data selects when training stops. It predicts benchmark dataset classes, not
diagnoses. Its normalized class scores are not calibrated clinical
probabilities. A coarse pixel-attribution overlay explains the baseline's
features; it is not a lesion map or probability heatmap. Upload screening uses
a validation-embedding distance check, prediction uncertainty, and simple
validation-profile checks for extreme brightness, contrast, and edge activity.
It is not a validated detector for arbitrary medical images.
Segmentation and the failure gallery remain unavailable until separately
connected.

On the official held-out test split, one run produced 181 true negatives, 53
false positives, 24 false negatives, and 366 true positives (624 images;
87.7% accuracy, 93.8% sensitivity, 77.4% specificity). This is a small
benchmark result, not evidence of clinical performance. The automated failure
gallery and human shortcut-cue annotations are not connected.

The Workspace preloads six PneumoniaMNIST training images, balanced between
the benchmark labels `normal` and `pneumonia`. Each is a pediatric chest X-ray,
not CT or MRI; each label is a dataset annotation, not a diagnosis.

If benchmark samples do not appear, the app shows the API URL and error. Check
that `EXPO_PUBLIC_API_URL` in `mobile/.env` is `http://127.0.0.1:8000` for a
browser running on the same computer. Restart the Python API after updating
project files, then tap **Retry loading demo cases**. The API serves each scan
image from its own PNG endpoint.
On first use, the API downloads and checksum-verifies an
approximately 214 MB dataset archive; keep that server terminal running and
allow the download to finish. The first analysis also trains the demo baseline
and prepares its validation reference.

The default browser API address is `http://127.0.0.1:8000`. The API allows
browser requests from `localhost:8081` and `127.0.0.1:8081`. If Expo prints a
different origin, set `SCANEXPLAIN_CORS_ORIGINS` to that exact origin before
starting the Python API, for example:

```sh
SCANEXPLAIN_CORS_ORIGINS=http://localhost:8081,http://127.0.0.1:8081 python -m scanexplain
```

## Demo sample dataset

The API downloads official PneumoniaMNIST MedMNIST+ 224×224 data on
request to `~/.cache/scanexplain/` on the Python server. Its 214 MB archive is
verified against the official MedMNIST v3 MD5 checksum. It is not copied into
the repository. License: CC BY 4.0; attribution is included in the app.

PneumoniaMNIST is a small, downsampled pediatric chest X-ray classification
benchmark. This simple baseline is not a clinically validated model. The
dataset has no segmentation masks, does not represent general emergency
imaging, and its maintainers explicitly state it is **not intended for
clinical use**. It cannot validate emergency diagnosis or generalize to
arbitrary medical uploads.

## Integration contract

Connect the team implementations with `scanexplain.ModelServices`:

```python
from scanexplain import ModelServices
from scanexplain.api import create_app

services = ModelServices(
    predict_fn=predict,   # RGB image -> {"label": str, "probs": {class: probability}}
    explain_fn=explain,   # RGB image -> explanation overlay
    segment_fn=segment,   # RGB image -> segmentation overlay
    embedding_fn=embed,   # optional upload-screening embedding
    ood_reference=reference,
    compare_fn=compare,   # (RGB image, prediction) -> (labeled image, label)
)
app = create_app(services, failure_report=test_report)
```

All components receive the same RGB-normalized image. Prediction probability
values must sum to one and include the predicted label. See
[the trust-layer guide](./docs/trust-layer.md).

## Project structure

- `mobile/` — Expo Go client.
- `scanexplain/api.py` — FastAPI upload, health, benchmark-sample, and failure-report endpoints.
- `scanexplain/pipeline.py` — shared preprocessing and model contracts.
- `scanexplain/dataset.py` — checksum-verified PneumoniaMNIST loader.
- `scanexplain/trust.py` — failure analysis and OOD-screening helpers.
- `tests/` — standard-library unit tests.

Run Python tests with `python -m unittest discover -s tests -v`.
