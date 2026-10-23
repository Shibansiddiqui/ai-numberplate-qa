import os
import hashlib
from datetime import datetime, timezone
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Import modular components
from database import fetch_events, insert_event, event_exists, update_correction
from vision import decode_image, image_quality, detect_license_plates, run_ocr, sha256_bytes
from copilot import answer_with_llm, deterministic_summary

st.set_page_config(page_title="AI Number Plate QA", page_icon="🚗", layout="wide")

st.title("🚗 AI Number Plate QA")

tab_analyze, tab_events, tab_copilot, tab_about = st.tabs(
    ["Analyze image", "Event history", "Operations copilot", "About"]
)

with tab_analyze:
    st.subheader("Analyze a parking image")
    uploaded = st.file_uploader("Upload JPG or PNG", type=["jpg", "jpeg", "png"], key="image_upload")
    notes = st.text_input("Optional operator note", placeholder="e.g. sample image from night-time test")
    
    if uploaded is not None:
        try:
            raw, bgr = decode_image(uploaded)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption=uploaded.name, use_container_width=True)
            detector_confidence = st.slider("Plate detector confidence threshold", 0.10, 0.90, 0.25, 0.05)
            
            if st.button("Run analysis", type="primary"):
                with st.spinner("Checking image quality, detecting plates, and running OCR..."):
                    metrics = image_quality(bgr)
                    digest = sha256_bytes(raw)
                    duplicate = event_exists(digest)
                    detections = detect_license_plates(bgr, confidence=detector_confidence)
                    annotated = bgr.copy()
                    plate_texts = []
                    ocr_confidences = []
                    
                    for index, detection in enumerate(detections, start=1):
                        x1, y1, x2, y2 = detection["box"]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
                        label = f"Plate {index}: {detection['confidence']:.2f}"
                        cv2.putText(annotated, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)
                        text, conf, status = run_ocr(detection["crop"])
                        if text:
                            plate_texts.append(text)
                        if conf is not None:
                            ocr_confidences.append(conf)
                            
                    ocr_text = " | ".join(plate_texts)
                    ocr_conf = round(float(np.mean(ocr_confidences)), 4) if ocr_confidences else None
                    
                    if not detections:
                        ocr_status = "NO_PLATE_DETECTED"
                    elif not ocr_text:
                        ocr_status = "PLATE_DETECTED_NO_TEXT"
                    elif ocr_conf is not None and ocr_conf < 0.55:
                        ocr_status = "OCR_LOW_CONFIDENCE"
                    else:
                        ocr_status = "OCR_TEXT_DETECTED"

                    review_needed = metrics["quality_status"] == "NEEDS_REVIEW"
                    if ocr_status in ("OCR_LOW_CONFIDENCE", "NO_PLATE_DETECTED", "PLATE_DETECTED_NO_TEXT", "OCR_UNAVAILABLE") or str(ocr_status).startswith("OCR_UNAVAILABLE"):
                        review_needed = True
                    if duplicate:
                        review_needed = True
                    if review_needed:
                        metrics["quality_status"] = "NEEDS_REVIEW"

                    event = {
                        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "filename": uploaded.name,
                        "image_sha256": digest,
                        "duplicate_in_db": duplicate,
                        "blur_score": metrics["blur_score"],
                        "brightness": metrics["brightness"],
                        "contrast": metrics["contrast"],
                        "quality_status": metrics["quality_status"],
                        "ocr_text": ocr_text,
                        "ocr_confidence": ocr_conf,
                        "ocr_status": ocr_status,
                        "operator_correction": "",
                        "notes": notes,
                    }
                    event_id = insert_event(event)
                    st.session_state["last_analysis"] = {
                        **event, **metrics, "id": event_id,
                        "plate_count": len(detections),
                        "annotated_rgb": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                        "plate_detections": [{"confidence": d["confidence"], "crop_rgb": cv2.cvtColor(d["crop"], cv2.COLOR_BGR2RGB)} for d in detections]
                    }

            last = st.session_state.get("last_analysis")
            if last and last.get("filename") == uploaded.name:
                st.divider()
                st.subheader(f"Analysis result — Event #{last['id']}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Blur score", f"{last['blur_score']:.1f}")
                c2.metric("Brightness", f"{last['brightness']:.1f}/255")
                c3.metric("Contrast", f"{last['contrast']:.1f}")
                st.write("**Review status:**", last["quality_status"])
                st.write("**Quality notes:**", ", ".join(last.get("quality_reasons", [])) or "No heuristic flags")
                st.write("**Detected plates:**", last.get("plate_count", 0))
                if last.get("annotated_rgb") is not None:
                    st.image(last["annotated_rgb"], caption="Detected plate regions", use_container_width=True)
                for i, plate in enumerate(last.get("plate_detections", []), start=1):
                    st.image(plate["crop_rgb"], caption=f"Plate crop {i} — detector confidence {plate['confidence']:.2%}", width=300)
                st.write("**OCR text:**", last["ocr_text"] or "No text extracted")
                st.write("**OCR confidence:**", f"{last['ocr_confidence']:.2%}" if last["ocr_confidence"] is not None else "Unavailable")
                st.write("**OCR status:**", last["ocr_status"])
                if last["duplicate_in_db"]:
                    st.warning("This exact file hash has appeared in a previous event.")
                st.info("The quality thresholds are illustrative heuristics. Validate them on a labeled dataset before drawing operational conclusions.")
        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")

with tab_events:
    st.subheader("Stored event history")
    events = fetch_events()
    if not events:
        st.info("No events yet. Analyze an image first, or load the optional sample records below.")
        if st.button("Load synthetic sample events"):
            samples = [
                {"filename":"sample_night_blur.jpg","blur_score":24.5,"brightness":32.0,"contrast":18.0,"quality_status":"NEEDS_REVIEW","ocr_text":"","ocr_confidence":None,"ocr_status":"NO_TEXT_DETECTED","duplicate_in_db":0,"notes":"Synthetic example: low light and blur"},
                {"filename":"sample_clear_crop.jpg","blur_score":145.2,"brightness":126.0,"contrast":52.0,"quality_status":"QUALITY_CHECK_PASSED","ocr_text":"DEMO 1234","ocr_confidence":0.91,"ocr_status":"OCR_TEXT_DETECTED","duplicate_in_db":0,"notes":"Synthetic example; OCR text is illustrative"},
                {"filename":"sample_glare.jpg","blur_score":92.0,"brightness":224.0,"contrast":22.0,"quality_status":"NEEDS_REVIEW","ocr_text":"D3M0","ocr_confidence":0.42,"ocr_status":"OCR_LOW_CONFIDENCE","duplicate_in_db":0,"notes":"Synthetic example: overexposure"},
            ]
            for s in samples:
                insert_event({
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "image_sha256": hashlib.sha256((s["filename"] + "-synthetic").encode()).hexdigest(),
                    "operator_correction": "",
                    **s,
                })
            st.success("Loaded synthetic event records.")
            st.rerun()
    else:
        df = pd.DataFrame(events)
        col1, col2 = st.columns(2)
        with col1:
            filter_status = st.selectbox("Filter by review status", ["All", "NEEDS_REVIEW", "QUALITY_CHECK_PASSED"])
        with col2:
            search = st.text_input("Search filename or OCR text")
        if filter_status != "All":
            df = df[df["quality_status"] == filter_status]
        if search.strip():
            mask = (
                df["filename"].astype(str).str.contains(search, case=False, na=False) |
                df["ocr_text"].astype(str).str.contains(search, case=False, na=False)
            )
            df = df[mask]
        display_cols = ["id", "created_at", "filename", "quality_status", "blur_score",
                        "brightness", "contrast", "ocr_text", "ocr_confidence", "ocr_status",
                        "duplicate_in_db", "operator_correction", "notes"]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
        st.download_button(
            "Download event history as CSV",
            data=df[display_cols].to_csv(index=False).encode("utf-8"),
            file_name="parkvision_events.csv",
            mime="text/csv"
        )
        st.divider()
        st.subheader("Human review / correction")
        ids = [int(x) for x in df["id"].tolist()]
        if ids:
            chosen_id = st.selectbox("Choose event ID", ids)
            chosen = next((e for e in events if e["id"] == chosen_id), None)
            with st.form("review_form"):
                correction = st.text_input("Corrected OCR text", value=(chosen or {}).get("operator_correction") or "")
                review_note = st.text_input("Review note", value=(chosen or {}).get("notes") or "")
                submitted = st.form_submit_button("Save review")
                if submitted:
                    update_correction(chosen_id, correction, review_note)
                    st.success(f"Saved review for event #{chosen_id}.")
                    st.rerun()

with tab_copilot:
    st.subheader("Ask the operations copilot")
    events = fetch_events()
    question = st.text_area(
        "Your question",
        placeholder="Which events need manual review, and what evidence supports that?",
        height=90
    )
    if st.button("Ask copilot", type="primary", key="ask_copilot"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Reviewing stored event records..."):
                answer, error = answer_with_llm(question.strip(), events)
                if answer:
                    st.markdown(answer)
                else:
                    st.info(deterministic_summary(question.strip(), events))
                    if error:
                        st.caption(error)

with tab_about:
    st.subheader("About ParkVision AI")
    st.write("A modular demo pipeline combining OpenCV heuristics, YOLOv8 plate detection, EasyOCR, SQLite persistence, and Gemini-powered operations copilot QA.")