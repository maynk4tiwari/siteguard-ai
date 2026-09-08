"""SiteGuard AI — FastAPI backend.

Browser -> FastAPI -> (vision_service, claude_service) -> Dashboard.
The Claude API key stays server-side only (.env). Never expose it to the frontend.
"""
import os
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from services import claude_service, demo_service, vision_service  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "uploads"
FRONTEND = ROOT / "frontend"
UPLOADS.mkdir(exist_ok=True)

MAX_FILE_MB = 15
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="SiteGuard AI", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# In-memory site state (MVP rule: no database).
STATE: dict = {"analysis": None, "alert_history": []}


class AssistantQuery(BaseModel):
    question: str


class ReportRequest(BaseModel):
    pass


def _current_analysis():
    if not STATE["analysis"]:
        raise HTTPException(
            status_code=404,
            detail="No site analysis yet. Upload an image or load the demo site first.",
        )
    return STATE["analysis"]


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "vision_ready": vision_service.is_available(),
        "claude_ready": claude_service.is_available(),
    }


@app.get("/api/demo")
def load_demo():
    """Load the clearly-labeled Prototype Demo Data and run Claude reasoning on it."""
    detections = demo_service.demo_detections()
    analysis = claude_service.analyze_site(detections)
    result = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "demo",
        "data_label": "Prototype Demo Data",
        "image_url": "/assets/demo-site.svg",
        "detections": detections,
        "analysis": analysis,
    }
    STATE["analysis"] = result
    STATE["alert_history"] = (analysis.get("alerts") or []) + STATE["alert_history"]
    return result


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Real pipeline: image -> YOLO detection -> structured data -> Claude reasoning."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Unsupported file type. Upload a JPG, PNG, or WEBP image.")
    raw = await file.read()
    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large. Maximum size is {MAX_FILE_MB} MB.")
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")

    ext = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    saved = UPLOADS / f"{uuid.uuid4().hex}{ext}"
    saved.write_bytes(raw)

    try:
        detections = vision_service.detect(str(saved))
    except vision_service.VisionUnavailable as e:
        raise HTTPException(
            503,
            "Computer-vision model is not available on this machine. "
            f"({e}) Fix: pip install ultralytics — or use Demo Mode.",
        )
    except Exception:
        import traceback
        traceback.print_exc()  # full details in the server terminal only
        raise HTTPException(500, "Computer-vision analysis failed for this image. Check the server terminal for details.")

    analysis = claude_service.analyze_site(detections)
    result = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "upload",
        "data_label": "Detected Observation (YOLOv8n) + AI Interpretation",
        "image_url": f"/api/uploads/{saved.name}",
        "detections": detections,
        "analysis": analysis,
    }
    STATE["analysis"] = result
    STATE["alert_history"] = (analysis.get("alerts") or []) + STATE["alert_history"]
    return result


@app.get("/api/uploads/{name}")
def get_upload(name: str):
    path = UPLOADS / os.path.basename(name)  # basename blocks path traversal
    if not path.exists():
        raise HTTPException(404, "File not found.")
    return FileResponse(path)


ALLOWED_VIDEO = {"video/mp4", "video/webm", "video/x-msvideo", "video/quicktime"}
MAX_VIDEO_MB = 80


@app.post("/api/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    """Video pipeline: sample frames -> YOLO per frame -> aggregate (real idle
    observation from stationary equipment) -> Claude reasoning."""
    if file.content_type not in ALLOWED_VIDEO:
        raise HTTPException(400, "Unsupported video type. Use MP4, WEBM, MOV, or AVI.")
    raw = await file.read()
    if len(raw) > MAX_VIDEO_MB * 1024 * 1024:
        raise HTTPException(400, f"Video too large. Maximum size is {MAX_VIDEO_MB} MB.")
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")

    ext = os.path.splitext(file.filename or "clip.mp4")[1] or ".mp4"
    saved = UPLOADS / f"{uuid.uuid4().hex}{ext}"
    saved.write_bytes(raw)
    keyframe = UPLOADS / f"{saved.stem}_key.jpg"

    try:
        detections = vision_service.detect_video(str(saved), str(keyframe))
    except vision_service.VisionUnavailable as e:
        raise HTTPException(503, f"Computer-vision model is not available. ({e}) Fix: pip install ultralytics — or use Demo Mode.")
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    analysis = claude_service.analyze_site(detections)
    result = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "video",
        "data_label": detections["data_label"] + " + AI Interpretation",
        "image_url": f"/api/uploads/{keyframe.name}",
        "detections": detections,
        "analysis": analysis,
    }
    STATE["analysis"] = result
    STATE["alert_history"] = (analysis.get("alerts") or []) + STATE["alert_history"]
    return result


@app.post("/api/analyze_frame")
async def analyze_frame(file: UploadFile = File(...)):
    """Fast live-camera path: one frame in, detections out. No Claude call per
    frame (that would be slow and expensive) and no state overwrite."""
    raw = await file.read()
    if not raw or len(raw) > 8 * 1024 * 1024:
        raise HTTPException(400, "Bad frame.")
    try:
        det = vision_service.detect_frame(raw)  # bytes in, no temp file at all
    except vision_service.VisionUnavailable as e:
        raise HTTPException(503, f"Computer-vision model is not available. ({e}) Fix: pip install ultralytics.")
    except RuntimeError as e:
        raise HTTPException(500, str(e))
    boxes = det["boxes"]
    return {
        "workers": det["workers"],
        "equipment": det["equipment"],
        "boxes": boxes,
        "proximity_observations": vision_service._proximity(boxes),
    }


@app.post("/api/analyze_live")
async def analyze_live(file: UploadFile = File(...), stats: str = Form(...)):
    """Turn a live-monitoring session into a full Claude analysis: the client
    sends one snapshot + its aggregated session stats."""
    import json as _json
    try:
        agg = _json.loads(stats)
    except Exception:
        raise HTTPException(400, "Bad session stats.")
    raw = await file.read()
    snap = UPLOADS / f"{uuid.uuid4().hex}_live.jpg"
    snap.write_bytes(raw)

    detections = {
        "mode": "live_camera",
        "data_label": f"Detected Observation (live camera · {int(agg.get('frames', 0))} frames analyzed)",
        "workers": int(agg.get("max_workers", 0)),
        "equipment_count": len(agg.get("equipment_types", [])),
        "equipment_observations": [
            {"type": t, "status": "observed", "confidence": 0.0} for t in agg.get("equipment_types", [])
        ],
        "ppe": {"evaluated": False, "note": "PPE detection not supported by the pretrained model in live mode."},
        "restricted_zone_events": {"evaluated": False},
        "boxes": agg.get("last_boxes", []),
        "proximity_observations": agg.get("proximity", []),
        "session": {"duration_s": agg.get("duration_s", 0), "frames": agg.get("frames", 0)},
    }
    analysis = claude_service.analyze_site(detections)
    result = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "live",
        "data_label": detections["data_label"] + " + AI Interpretation",
        "image_url": f"/api/uploads/{snap.name}",
        "detections": detections,
        "analysis": analysis,
    }
    STATE["analysis"] = result
    STATE["alert_history"] = (analysis.get("alerts") or []) + STATE["alert_history"]
    return result


@app.get("/api/state")
def get_state():
    return {"analysis": STATE["analysis"], "alert_history": STATE["alert_history"][:50]}


@app.post("/api/check_ppe")
def check_ppe():
    """Claude Vision PPE check (helmets & boots) on the current analysis image.
    Merges the result into the site state and re-runs Claude's risk reasoning so
    alerts, score, and reports include the PPE findings."""
    analysis = _current_analysis()
    if analysis["source"] == "demo":
        raise HTTPException(400, "Demo Mode already includes simulated PPE data. Upload an image, video, or use the live monitor for a real PPE vision check.")

    img_name = os.path.basename(analysis["image_url"])
    img_path = UPLOADS / img_name
    if not img_path.exists():
        raise HTTPException(404, "The analyzed image is no longer on disk — re-run the analysis first.")
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}.get(img_path.suffix.lower())
    if not media:
        raise HTTPException(400, "PPE vision check supports JPG/PNG/WEBP frames.")

    try:
        ppe = claude_service.analyze_ppe(img_path.read_bytes(), media, analysis["detections"])
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception:
        import traceback; traceback.print_exc()
        raise HTTPException(500, "PPE vision check failed. Check the server terminal for details.")

    # Recolor the exact worker boxes: missing PPE -> red "violation" box with a
    # specific label; unclear -> amber "unclear" box. Worn-everything stays green.
    worker_boxes = [b for b in analysis["detections"].get("boxes", []) if b.get("kind") in ("worker", "violation", "unclear")]
    for wkr in ppe.get("per_worker", []):
        i = wkr.get("index")
        if not isinstance(i, int) or i < 0 or i >= len(worker_boxes):
            continue
        box = worker_boxes[i]
        missing = [item for item in ("helmet", "boots") if wkr.get(item) == "missing"]
        unclear = [item for item in ("helmet", "boots") if wkr.get(item) == "unclear"]
        if missing:
            box["kind"] = "violation"
            box["label"] = "WORKER · ⚠ NO " + "+".join(m.upper() for m in missing)
        elif unclear:
            box["kind"] = "unclear"
            box["label"] = "WORKER · PPE UNCLEAR (" + "/".join(unclear) + ")"
        else:
            box["kind"] = "worker"
            box["label"] = "WORKER · PPE OK"

    # Merge into detections, honestly labeled, then let Claude re-prioritize.
    analysis["detections"]["ppe"] = {
        "evaluated": True,
        "method": ppe["source"],
        "workers_checked": ppe.get("workers_seen", 0),
        "helmet_violations": ppe.get("helmet", {}).get("missing", 0),
        "helmets_worn": ppe.get("helmet", {}).get("wearing", 0),
        "helmet_unclear": ppe.get("helmet", {}).get("unclear", 0),
        "boot_violations": ppe.get("boots", {}).get("missing", 0),
        "boots_worn": ppe.get("boots", {}).get("wearing", 0),
        "boots_unclear": ppe.get("boots", {}).get("unclear", 0),
        "vest_violations": 0,
        "notes": ppe.get("notes", []),
        "per_worker": ppe.get("per_worker", []),
    }
    analysis["analysis"] = claude_service.analyze_site(analysis["detections"])
    analysis["data_label"] = analysis["data_label"].split(" + ")[0] + " + AI Vision PPE Check + AI Interpretation"
    STATE["analysis"] = analysis
    STATE["alert_history"] = (analysis["analysis"].get("alerts") or []) + STATE["alert_history"]
    return {"ppe": ppe, "bundle": analysis}


@app.post("/api/report")
def generate_report():
    analysis = _current_analysis()
    report = claude_service.generate_report(analysis)
    return {"report": report, "data_label": analysis["data_label"]}


@app.post("/api/assistant")
def assistant(query: AssistantQuery):
    if not query.question.strip():
        raise HTTPException(400, "Please type a question first.")
    analysis = _current_analysis()
    answer = claude_service.assistant_answer(query.question.strip(), analysis)
    return {"answer": answer}


@app.exception_handler(Exception)
async def unhandled(request, exc):
    # Friendly errors only in the UI — full traceback goes to the server terminal.
    import traceback
    traceback.print_exception(exc)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server. Check the server terminal for details."})


# Serve the frontend last so /api/* wins.
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")
