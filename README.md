# ParkVision AI

Independent portfolio prototype inspired by public smart-parking use cases. It is not an official PARKONIC product and does not use PARKONIC internal data.

## Features

- OpenCV image-quality heuristics (blur, brightness, contrast)
- OCR using EasyOCR on an uploaded image or manually cropped plate image
- Exact-file duplicate detection using SHA-256
- SQLite event history, operator correction, CSV export
- Optional LLM operations copilot grounded in stored event records
- Streamlit dashboard

## Important limitation

This 3-day MVP does **not** include a license-plate detector. For best OCR results, upload a close-up plate crop. Image-quality thresholds are illustrative and not validated for operational use.

## Run locally

Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first OCR call downloads model weights and may take a while.

## Enable the LLM copilot

Set `OPENAI_API_KEY` as an environment variable. Optionally set `OPENAI_MODEL` (default `gpt-4o-mini`).

macOS/Linux:
```bash
export OPENAI_API_KEY="your-key"
streamlit run app.py
```

PowerShell:
```powershell
$env:OPENAI_API_KEY="your-key"
streamlit run app.py
```

Never commit API keys. When the LLM is enabled, the question and selected event fields are sent to the configured provider. Use synthetic demo records only.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open https://share.streamlit.io and choose **Create app**.
3. Select your repository, branch, and `app.py`.
4. In app settings, add `OPENAI_API_KEY` and optionally `OPENAI_MODEL` under Secrets, using TOML syntax:

```toml
OPENAI_API_KEY = "your-key"
OPENAI_MODEL = "gpt-4o-mini"
```

5. Deploy and test from a fresh session.

Free cloud filesystems may not persist SQLite data across restarts. This is acceptable for a demo; use managed PostgreSQL/object storage for a production design.

## Suggested 3-day plan

- Day 1: run app, test image metrics, inspect OCR output on plate crops
- Day 2: event review workflow, sample records, LLM copilot, tests
- Day 3: deploy, benchmark labeled examples, write limitations and demo script

## Evaluation

Build a small, licensed/authorized dataset with human labels. Report:
- OCR exact-match accuracy on readable plate crops
- Review-flag precision and recall
- Duplicate detection results
- Median processing latency
- Failure cases by lighting/blur

Do not claim metrics until you have measured them on a labeled dataset.


## Automatic license-plate detection add-on

This version includes an experimental YOLOv8 plate detector. On the first analysis, it downloads `best.pt` from the community Hugging Face repository `Koushim/yolov8-license-plate-detection`; internet access is required. It detects plate regions in the full image, draws boxes, crops them, and sends each crop to EasyOCR. The model card lists an MIT license for its weights; check the Ultralytics package's current license and your intended use before redistribution/commercial deployment.

This is a portfolio prototype, not a validated ANPR system. Test with authorized images representative of your target camera, region, lighting, plate styles, and image sizes. The model may miss small plates or detect false positives. Do not use results alone for enforcement or access decisions.
