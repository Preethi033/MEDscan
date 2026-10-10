# Failure analysis and trust layer

MedScan is an educational demo for supported sample images. It is not a
medical device and is **not for diagnosis, treatment decisions, or emergency
care**. A model prediction, confidence score, similarity score, or visual
explanation must not be treated as clinical advice.

## Expo Go and API

The mobile UI is under `mobile/`; Expo Go is the app container. Python model
code runs separately in the FastAPI backend:

```sh
# Terminal 1: from the repository root
python -m pip install '.[api]'
python -m scanexplain

# Terminal 2
cd mobile
cp .env.example .env
# Set EXPO_PUBLIC_API_URL to http://<computer-LAN-IP>:8000 in .env
npm install
npx expo start
```

The computer and phone must be on the same trusted LAN. The backend binds to
`0.0.0.0` for Expo Go connectivity and has no authentication; never expose it
to the public internet or an untrusted network.

For a browser-only demo on the same computer, run `cd mobile && npm install &&
npx expo start --web`, then open the local URL Expo prints. The web client
defaults to the Python API at `http://127.0.0.1:8000`. API CORS allows the
standard `localhost:8081` and `127.0.0.1:8081` Expo web origins; set
`SCANEXPLAIN_CORS_ORIGINS` to comma-separated exact origins if the dev server
uses a different port or host.

The default API connects a small PneumoniaMNIST logistic baseline for the
benchmark classes, a coarse linear-feature attribution overlay, and a
validation-embedding OOD screen. It validates JPEG, PNG, and WebP uploads
(12 MB maximum; 4096-pixel maximum side) and normalizes decoded images through
`preprocess_image`. Segmentation and a labeled failure report remain
unconfigured. Invalid model outputs and operational errors are not replaced
with success-shaped results. The classifier is trained on first analysis;
startup and sample browsing do not trigger model fitting.

Connect implementations with `ModelServices`:

```python
from scanexplain import ModelServices
from scanexplain.api import create_app

services = ModelServices(
    predict_fn=predict,       # {"label": str, "probs": {class_name: probability}}
    explain_fn=explain,       # explanation overlay image
    segment_fn=segment,       # segmentation overlay image
    embedding_fn=embed,       # optional; required together with ood_reference
    ood_reference=reference,  # optional fitted EmbeddingReference
    compare_fn=compare,       # (uploaded_image, prediction) -> (image, label)
)
app = create_app(services, failure_report=test_report)
```

All callbacks receive an RGB image from the shared preprocessing function.
Prediction probabilities must be finite, between zero and one, sum to one, and
include the predicted label. Overlays must be image objects encodable as PNG.

## Benchmark sample data

The API downloads MedMNIST PneumoniaMNIST+ at 224×224 from the official Zenodo
distribution only when requested. The verified archive is
cached at `~/.cache/scanexplain/`; it is not committed to the repository. Its
current archive is approximately 214 MB. Official dataset license: CC BY 4.0.
Attribution and dataset/source citations are included in the API response and
mobile UI.

The Demo Cases tab assigns six training examples to fictional names. Search
results show names only; the details panel shows one selected case. Display
IDs and scan dates/times are synthetic UI metadata, not patient identifiers or
image acquisition details. Uploaded images are not associated with fictional
names. The displayed class label is a dataset annotation, not a diagnosis.

The baseline downsamples images to 16×16 grayscale features and trains a
logistic classifier using only the benchmark training split. The validation
split selects the stopping epoch and provides reference embeddings for upload
screening. Its normalized scores are not calibrated probabilities of a
medical condition, and its interpolated attribution overlay is not a lesion
map.

One held-out test-split run produced 181 true negatives, 53 false positives,
24 false negatives, and 366 true positives across 624 images (87.7% accuracy,
93.8% sensitivity, 77.4% specificity). This small benchmark result is not
clinical evidence. The automated failure gallery and human shortcut-cue
annotations remain unconnected.

The dataset is based on pediatric chest X-rays, downsampled to 224×224, and
provides classification labels only—not segmentation masks. MedMNIST's
maintainers explicitly state that the dataset is **not intended for clinical
use**. It is an educational benchmark, not evidence of emergency diagnostic
performance or support for arbitrary uploads.

## Evaluating model failures

`scanexplain.trust.analyze_predictions` accepts labeled predictions from a
separate evaluation/test split and returns counts plus a failure gallery. It
includes misclassifications, predictions below a configurable confidence
threshold, and examples with evaluator-annotated shortcut cues such as
artifacts, markers, or borders. The generated note describes observable
evidence; it does not claim to know why the model made its prediction.

Shortcut cues must be identified by a human or a separately validated
detector. The helper does not inspect pixels or infer that a marker or border
caused an error. Keep the test split separate from training and threshold
selection. `create_app(..., failure_report=report)` exposes a report at
`GET /api/failures`; until supplied, the app explicitly reports that no
evaluation report is connected.

## Upload screening

`scanexplain.trust.check_ood` reports prediction confidence and normalized
entropy. If given an image embedding and an `EmbeddingReference`, it also
compares the embedding to supported validation embeddings using cosine
distance. Fit the reference with representative, held-out supported samples
from the exact model/preprocessing pipeline used for uploads.

- `ood_suspected`: embedding distance exceeds the reference threshold.
- `uncertain`: prediction confidence or entropy crosses its review threshold.
- `in_distribution_like`: supplied embedding is within the reference threshold
  and uncertainty thresholds did not trigger. This is not proof that an image
  is supported or safe.
- `cannot_assess`: there is no embedding reference and uncertainty checks did
  not trigger. The app should not treat this as an in-distribution result.

These are screening signals, not a validated medical-image detector. A
high-confidence prediction can still be wrong, and an unsupported image can
look similar to supported embeddings. These checks must not be used to decide
whether an emergency image is safe to assess.
The demo baseline also flags extreme brightness, contrast, and edge-activity
profiles relative to its validation split. These simple heuristics can miss
unsupported images and can flag unusual but valid scans; they do not establish
that an upload is a chest X-ray.

## Known limitations

- The bundled classifier is a simple educational logistic baseline, not a
  clinically validated or calibrated model.
- No segmentation model or labeled failure gallery is bundled.
- Confidence and entropy depend on model calibration and can be misleading.
- Embedding-distance screening depends on representative reference data,
  consistent preprocessing, and a chosen quantile; it may miss OOD inputs or
  flag valid but unusual cases.
- Shortcut-cue annotations are not causal explanations. Attribution maps can
  also be unstable or misleading and need their own validation.
- Dataset labeling errors, demographic under-representation, acquisition
  differences, image quality, and distribution shift can change performance.
- This prototype does not replace expert review, clinical validation, privacy
  controls, or regulatory review.
