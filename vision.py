import hashlib
import cv2
import numpy as np
import streamlit as st

def decode_image(uploaded_file):
    raw = uploaded_file.getvalue()
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode this image. Try a JPG or PNG.")
    return raw, bgr

def image_quality(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    reasons = []
    if blur_score < 80:
        reasons.append("Possible blur (low Laplacian variance)")
    if brightness < 45:
        reasons.append("Image may be too dark")
    elif brightness > 210:
        reasons.append("Image may be overexposed")
    if contrast < 25:
        reasons.append("Low contrast")

    status = "NEEDS_REVIEW" if reasons else "QUALITY_CHECK_PASSED"
    return {
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "quality_status": status,
        "quality_reasons": reasons,
        "width": int(bgr.shape[1]),
        "height": int(bgr.shape[0]),
    }

@st.cache_resource(show_spinner="Loading license-plate detector (first run downloads model weights)...")
def get_plate_detector():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO
    weights = hf_hub_download(
        repo_id="Koushim/yolov8-license-plate-detection",
        filename="best.pt",
    )
    return YOLO(weights)

def detect_license_plates(bgr, confidence=0.25):
    detector = get_plate_detector()
    result = detector.predict(source=bgr, conf=confidence, verbose=False)[0]
    detections = []
    if result.boxes is None:
        return detections
    height, width = bgr.shape[:2]
    for box in result.boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = bgr[y1:y2, x1:x2].copy()
        if crop.size:
            detections.append({
                "box": (x1, y1, x2, y2),
                "confidence": float(box.conf[0]),
                "crop": crop,
            })
    return detections

@st.cache_resource(show_spinner="Loading OCR model (first run may take a while)...")
def get_ocr_reader():
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)

def run_ocr(bgr):
    try:
        reader = get_ocr_reader()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = reader.readtext(rgb, detail=1, paragraph=False)
        if not results:
            return "", None, "NO_TEXT_DETECTED"
        pieces = []
        confidences = []
        for item in results:
            if len(item) >= 3:
                text, conf = str(item[1]).strip(), float(item[2])
                if text:
                    pieces.append(text)
                    confidences.append(conf)
        combined = " ".join(pieces)
        avg_conf = round(float(np.mean(confidences)), 4) if confidences else None
        status = "OCR_LOW_CONFIDENCE" if avg_conf is not None and avg_conf < 0.55 else "OCR_TEXT_DETECTED"
        return combined, avg_conf, status
    except Exception as exc:
        return "", None, f"OCR_UNAVAILABLE: {type(exc).__name__}"

def sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()