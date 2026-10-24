AI Numberplate QA

Features

OpenCV image-quality heuristics (blur, brightness, contrast)

OCR using EasyOCR on an uploaded image or manually cropped plate image

Automatic YOLOv8 license-plate region detection and cropping

Exact-file duplicate detection using SHA-256

SQLite event history, operator correction, CSV export

Optional Google Gemini operations copilot grounded in stored event records

Streamlit dashboard

Run locally

Python 3.10 or 3.11 recommended.

python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py


The first OCR and plate detector run will download model weights and may take a while.

Enable the Gemini LLM copilot

Set GEMINI_API_KEY as an environment variable. Optionally set LLM_MODEL (default gemini-2.5-flash).

macOS/Linux:

export GEMINI_API_KEY="your-key"
streamlit run app.py


PowerShell:

$env:GEMINI_API_KEY="your-key"
streamlit run app.py


Deploy to Streamlit Community Cloud

Push this folder to a GitHub repository.

Open https://share.streamlit.io and choose Create app.

Select your repository, branch, and app.py.

In app settings, add GEMINI_API_KEY and optionally LLM_MODEL under Secrets, using TOML syntax:

GEMINI_API_KEY = "your-key"
LLM_MODEL = "gemini-2.5-flash"


Automatic license-plate detection add-on

This version includes an experimental YOLOv8 plate detector. On the first analysis, it downloads best.pt from the community Hugging Face repository Koushim/yolov8-license-plate-detection; internet access is required. It detects plate regions in the full image, draws boxes, crops them, and sends each crop to EasyOCR. The model card lists an MIT license for its weights; check the Ultralytics package's current license and your intended use before redistribution/commercial deployment.