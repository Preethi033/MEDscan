# ScanExplain — Expo Go research demo

ScanExplain is a research prototype for examining medical-imaging model
behavior. It is **not for diagnosis, treatment decisions, emergency triage, or
delaying care**. Outputs are not clinical advice.

## Open it on Expo Go

You need Node.js/npm on the development computer and Expo Go on your iOS or
Android phone. Both devices must be on a trusted local network.

1. Start the Python API from the repository root:

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install '.[api]'
   python -m scanexplain
   ```

   The API listens on port `8000` on the local network so the phone can reach
   it. Do not expose it to a public network.

2. Find your computer's LAN IP address. On many Macs this is
   `ipconfig getifaddr en0`; otherwise find the Wi-Fi IPv4 address in Network
   settings.

3. Configure the Expo app:

   ```sh
   cd mobile
   cp .env.example .env
   ```

   Edit `.env` and replace `192.168.1.10` with the computer's LAN IP.

4. Start Expo and scan its QR code with Expo Go:

   ```sh
   npm install
   npx expo start
   ```

The repo targets Expo SDK 57. The app has image upload, API connection status,
prediction/explanation/segmentation placeholders, dataset samples, a failure
gallery, and a limitations view. Model components and test-set reports are
injected through the Python API and are clearly marked unavailable until
connected.

## Open it in a web browser instead

The Expo client also supports web, which avoids Expo Go and QR scanning:

1. Start the Python API in one terminal using the instructions above.
2. In a second terminal:

   ```sh
   cd /Users/seven/Documents/GitHub/MEDscan/mobile
   npm install
   npx expo start --web
   ```

3. Open the local URL Expo prints (usually `http://localhost:8081`) in your
   computer's browser.

The default browser API address is `http://127.0.0.1:8000`. The API allows
browser requests from `localhost:8081` and `127.0.0.1:8081`. If Expo prints a
different origin, set `SCANEXPLAIN_CORS_ORIGINS` to that exact origin before
starting the Python API, for example:

```sh
SCANEXPLAIN_CORS_ORIGINS=http://localhost:8081,http://127.0.0.1:8081 python -m scanexplain
```

Use LAN mode only if you want to open it on another device; the Expo Go and
browser setup still needs the Python API reachable from that device.

## Research dataset

The Samples view downloads official PneumoniaMNIST MedMNIST+ 224×224 data on
request to `~/.cache/scanexplain/` on the Python server. Its 214 MB archive is
verified against the official MedMNIST v3 MD5 checksum. It is not copied into
the repository. License: CC BY 4.0; attribution is included in the app.

PneumoniaMNIST is a small, downsampled pediatric chest X-ray classification
benchmark. It has no segmentation masks, does not represent general emergency
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
