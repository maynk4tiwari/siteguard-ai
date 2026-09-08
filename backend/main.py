"""
SiteGuard AI — FastAPI Backend

Architecture:
    Browser / Netlify Frontend
            ↓
        FastAPI Backend
            ↓
    ┌───────────────────────┐
    │ YOLO Vision Service   │
    │ Claude AI Service     │
    │ Demo Service          │
    └───────────────────────┘

The Anthropic API key stays on the server.
NEVER expose ANTHROPIC_API_KEY to the frontend.
"""

import os
import time
import uuid
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ============================================================
# ENVIRONMENT
# ============================================================

# Project root:
# siteguard-ai/
#
# Backend:
# siteguard-ai/backend/main.py
#
# Therefore parent.parent = siteguard-ai/
ROOT = Path(__file__).resolve().parent.parent

# Load .env from project root
ENV_FILE = ROOT / ".env"

load_dotenv(ENV_FILE)
load_dotenv()  # Also support environment variables from Render/local shell


# ============================================================
# API KEY
# ============================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

if ANTHROPIC_API_KEY and not ANTHROPIC_API_KEY.startswith(
    "sk-ant-placeholder"
):
    print(
        "[SiteGuard] Claude API key: LOADED "
        f"(...{ANTHROPIC_API_KEY[-4:]})"
    )
else:
    print("[SiteGuard] Claude API key: MISSING")
    print(
        "[SiteGuard] Claude features will use the fallback logic "
        "if supported by claude_service."
    )


# ============================================================
# SERVICES
# ============================================================

# Import after environment variables have been loaded.
#
# IMPORTANT:
# services must be available at:
#
# backend/
# ├── main.py
# └── services/
#
from services import claude_service, demo_service, vision_service


# ============================================================
# PATHS
# ============================================================

UPLOADS = ROOT / "uploads"
FRONTEND = ROOT / "frontend"

# Create uploads directory if it doesn't exist
UPLOADS.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FILE_MB = 15

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

ALLOWED_VIDEO_TYPES = {
    "video/mp4",
    "video/webm",
    "video/x-msvideo",
    "video/quicktime",
}

MAX_VIDEO_MB = 80


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="SiteGuard AI",
    description=(
        "AI-powered construction site safety and intelligence "
        "backend using YOLO computer vision and Claude AI."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

# Your Netlify frontend:
# https://siteguard-ai.netlify.app
#
# Local development is also allowed.

ALLOWED_ORIGINS = [
    "https://siteguard-ai.netlify.app",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# IN-MEMORY STATE
# ============================================================

# MVP does not use a database.
#
# WARNING:
# Render's free service can restart/sleep, so this state is
# temporary and can be lost after restart.

STATE = {
    "analysis": None,
    "alert_history": [],
}


# ============================================================
# REQUEST MODELS
# ============================================================

class AssistantQuery(BaseModel):
    question: str


class ReportRequest(BaseModel):
    """
    Currently no fields are required.
    Kept for compatibility with the frontend/API design.
    """

    pass


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _current_analysis():
    """
    Return the latest analysis.

    Raises:
        HTTPException:
            If no analysis exists.
    """

    if not STATE["analysis"]:
        raise HTTPException(
            status_code=404,
            detail=(
                "No site analysis yet. "
                "Upload an image or load the demo site first."
            ),
        )

    return STATE["analysis"]


def _save_upload(raw: bytes, filename: str, default_ext: str):
    """
    Save uploaded file safely using a random UUID filename.
    """

    original_ext = os.path.splitext(filename or "")[1].lower()

    if not original_ext:
        original_ext = default_ext

    # Only keep a simple extension.
    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".mp4",
        ".webm",
        ".mov",
        ".avi",
    }

    if original_ext not in allowed_extensions:
        original_ext = default_ext

    saved = UPLOADS / f"{uuid.uuid4().hex}{original_ext}"

    saved.write_bytes(raw)

    return saved


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    """
    Backend health/status endpoint.

    Render can use:
        /api/health
    as its health check path.
    """

    try:
        vision_ready = vision_service.is_available()
    except Exception:
        vision_ready = False

    try:
        claude_ready = claude_service.is_available()
    except Exception:
        claude_ready = False

    return {
        "status": "ok",
        "service": "SiteGuard AI",
        "version": "1.0.0",
        "vision_ready": vision_ready,
        "claude_ready": claude_ready,
        "environment": "production"
        if os.getenv("RENDER")
        else "development",
    }


# ============================================================
# ROOT API INFORMATION
# ============================================================

@app.get("/api")
def api_info():
    """
    Simple API information endpoint.
    """

    return {
        "name": "SiteGuard AI",
        "status": "online",
        "message": "SiteGuard AI backend is running.",
        "health": "/api/health",
        "docs": "/docs",
    }


# ============================================================
# DEMO MODE
# ============================================================

@app.get("/api/demo")
def load_demo():
    """
    Load clearly-labelled prototype demo data.

    Demo data is generated by demo_service and then
    interpreted by Claude/fallback logic.
    """

    try:
        detections = demo_service.demo_detections()

        analysis = claude_service.analyze_site(
            detections
        )

        result = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "source": "demo",
            "data_label": "Prototype Demo Data",
            "image_url": "/assets/demo-site.svg",
            "detections": detections,
            "analysis": analysis,
        }

        STATE["analysis"] = result

        STATE["alert_history"] = (
            analysis.get("alerts") or []
        ) + STATE["alert_history"]

        return result

    except Exception:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to load demo data. "
                "Check the backend logs."
            ),
        )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Main image-analysis pipeline:

        Uploaded image
                ↓
            YOLO detection
                ↓
        Structured detections
                ↓
            Claude AI
                ↓
        Safety analysis
    """

    # --------------------------------------------------------
    # Validate content type
    # --------------------------------------------------------

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file type. "
                "Upload a JPG, PNG, or WEBP image."
            ),
        )

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    # --------------------------------------------------------
    # Check size
    # --------------------------------------------------------

    if len(raw) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File too large. "
                f"Maximum size is {MAX_FILE_MB} MB."
            ),
        )

    # --------------------------------------------------------
    # Save file
    # --------------------------------------------------------

    saved = _save_upload(
        raw,
        file.filename or "upload.jpg",
        ".jpg",
    )

    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    try:

        detections = vision_service.detect(
            str(saved)
        )

    except vision_service.VisionUnavailable as e:

        raise HTTPException(
            status_code=503,
            detail=(
                "Computer-vision model is not available "
                f"on this server. ({e})"
            ),
        )

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Computer-vision analysis failed "
                "for this image. Check the backend logs."
            ),
        )

    # --------------------------------------------------------
    # Claude AI
    # --------------------------------------------------------

    try:

        analysis = claude_service.analyze_site(
            detections
        )

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "AI site analysis failed. "
                "Check the backend logs."
            ),
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "source": "upload",
        "data_label": (
            "Detected Observation (YOLOv8n) "
            "+ AI Interpretation"
        ),
        "image_url": (
            f"/api/uploads/{saved.name}"
        ),
        "detections": detections,
        "analysis": analysis,
    }

    # Save current analysis
    STATE["analysis"] = result

    # Add alerts to history
    STATE["alert_history"] = (
        analysis.get("alerts") or []
    ) + STATE["alert_history"]

    return result


# ============================================================
# SERVE UPLOADED FILE
# ============================================================

@app.get("/api/uploads/{name}")
def get_upload(name: str):
    """
    Return an uploaded image/video safely.

    basename() prevents path traversal.
    """

    safe_name = os.path.basename(name)

    path = UPLOADS / safe_name

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="File not found.",
        )

    return FileResponse(path)


# ============================================================
# VIDEO ANALYSIS
# ============================================================

@app.post("/api/analyze_video")
async def analyze_video(
    file: UploadFile = File(...)
):
    """
    Video pipeline:

        Video
          ↓
      Sample frames
          ↓
       YOLO
          ↓
      Aggregate
          ↓
      Claude AI
    """

    # --------------------------------------------------------
    # Validate video type
    # --------------------------------------------------------

    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video type. "
                "Use MP4, WEBM, MOV, or AVI."
            ),
        )

    # --------------------------------------------------------
    # Read video
    # --------------------------------------------------------

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="The uploaded video is empty.",
        )

    # --------------------------------------------------------
    # Size check
    # --------------------------------------------------------

    if len(raw) > MAX_VIDEO_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Video too large. "
                f"Maximum size is {MAX_VIDEO_MB} MB."
            ),
        )

    # --------------------------------------------------------
    # Save video
    # --------------------------------------------------------

    saved = _save_upload(
        raw,
        file.filename or "clip.mp4",
        ".mp4",
    )

    keyframe = UPLOADS / (
        f"{saved.stem}_key.jpg"
    )

    # --------------------------------------------------------
    # Vision processing
    # --------------------------------------------------------

    try:

        detections = vision_service.detect_video(
            str(saved),
            str(keyframe),
        )

    except vision_service.VisionUnavailable as e:

        raise HTTPException(
            status_code=503,
            detail=(
                "Computer-vision model is not available. "
                f"({e})"
            ),
        )

    except RuntimeError as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Video computer-vision analysis failed. "
                "Check the backend logs."
            ),
        )

    # --------------------------------------------------------
    # Claude
    # --------------------------------------------------------

    try:

        analysis = claude_service.analyze_site(
            detections
        )

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "AI video analysis failed. "
                "Check the backend logs."
            ),
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "source": "video",
        "data_label": (
            detections["data_label"]
            + " + AI Interpretation"
        ),
        "image_url": (
            f"/api/uploads/{keyframe.name}"
        ),
        "detections": detections,
        "analysis": analysis,
    }

    STATE["analysis"] = result

    STATE["alert_history"] = (
        analysis.get("alerts") or []
    ) + STATE["alert_history"]

    return result


# ============================================================
# LIVE FRAME ANALYSIS
# ============================================================

@app.post("/api/analyze_frame")
async def analyze_frame(
    file: UploadFile = File(...)
):
    """
    Fast live-camera path.

    One frame is sent to YOLO.

    Claude is NOT called for every frame because
    that would be slow and expensive.
    """

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Bad frame.",
        )

    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Frame is too large. Maximum size is 8 MB.",
        )

    try:

        det = vision_service.detect_frame(
            raw
        )

    except vision_service.VisionUnavailable as e:

        raise HTTPException(
            status_code=503,
            detail=(
                "Computer-vision model is not available. "
                f"({e})"
            ),
        )

    except RuntimeError as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Live frame analysis failed."
            ),
        )

    boxes = det.get("boxes", [])

    return {
        "workers": det.get("workers", 0),
        "equipment": det.get("equipment", 0),
        "boxes": boxes,
        "proximity_observations": (
            vision_service._proximity(boxes)
        ),
    }


# ============================================================
# LIVE SESSION ANALYSIS
# ============================================================

@app.post("/api/analyze_live")
async def analyze_live(
    file: UploadFile = File(...),
    stats: str = Form(...),
):
    """
    Turn a live-monitoring session into a complete
    Claude analysis.

    The frontend sends:
        1. Snapshot image
        2. Aggregated monitoring statistics
    """

    import json

    # --------------------------------------------------------
    # Parse stats
    # --------------------------------------------------------

    try:

        agg = json.loads(stats)

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Bad session stats.",
        )

    # --------------------------------------------------------
    # Read snapshot
    # --------------------------------------------------------

    raw = await file.read()

    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Live snapshot is empty.",
        )

    if len(raw) > 8 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="Live snapshot is too large.",
        )

    # --------------------------------------------------------
    # Save snapshot
    # --------------------------------------------------------

    snap = UPLOADS / (
        f"{uuid.uuid4().hex}_live.jpg"
    )

    snap.write_bytes(raw)

    # --------------------------------------------------------
    # Create aggregated detections
    # --------------------------------------------------------

    equipment_types = agg.get(
        "equipment_types",
        [],
    )

    detections = {
        "mode": "live_camera",

        "data_label": (
            "Detected Observation "
            f"(live camera · "
            f"{int(agg.get('frames', 0))} "
            "frames analyzed)"
        ),

        "workers": int(
            agg.get("max_workers", 0)
        ),

        "equipment_count": len(
            equipment_types
        ),

        "equipment_observations": [
            {
                "type": equipment_type,
                "status": "observed",
                "confidence": 0.0,
            }
            for equipment_type
            in equipment_types
        ],

        "ppe": {
            "evaluated": False,
            "note": (
                "PPE detection is not supported "
                "by the pretrained model in live mode."
            ),
        },

        "restricted_zone_events": {
            "evaluated": False
        },

        "boxes": agg.get(
            "last_boxes",
            [],
        ),

        "proximity_observations": agg.get(
            "proximity",
            [],
        ),

        "session": {
            "duration_s": agg.get(
                "duration_s",
                0,
            ),
            "frames": agg.get(
                "frames",
                0,
            ),
        },
    }

    # --------------------------------------------------------
    # Claude
    # --------------------------------------------------------

    try:

        analysis = claude_service.analyze_site(
            detections
        )

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Live AI analysis failed. "
                "Check the backend logs."
            ),
        )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    result = {
        "id": str(uuid.uuid4())[:8],

        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "source": "live",

        "data_label": (
            detections["data_label"]
            + " + AI Interpretation"
        ),

        "image_url": (
            f"/api/uploads/{snap.name}"
        ),

        "detections": detections,

        "analysis": analysis,
    }

    STATE["analysis"] = result

    STATE["alert_history"] = (
        analysis.get("alerts") or []
    ) + STATE["alert_history"]

    return result


# ============================================================
# CURRENT STATE
# ============================================================

@app.get("/api/state")
def get_state():
    """
    Return current site analysis and recent alerts.
    """

    return {
        "analysis": STATE["analysis"],
        "alert_history": (
            STATE["alert_history"][:50]
        ),
    }


# ============================================================
# PPE CHECK
# ============================================================

@app.post("/api/check_ppe")
def check_ppe():
    """
    Run Claude Vision PPE analysis on the current image.

    Supported:
        - Helmet
        - Boots

    The result is merged into the current site state.
    """

    analysis = _current_analysis()

    # --------------------------------------------------------
    # Demo mode
    # --------------------------------------------------------

    if analysis["source"] == "demo":

        raise HTTPException(
            status_code=400,
            detail=(
                "Demo Mode already includes simulated PPE data. "
                "Upload an image, video, or use the live monitor "
                "for a real PPE vision check."
            ),
        )

    # --------------------------------------------------------
    # Get analyzed image
    # --------------------------------------------------------

    image_url = analysis.get(
        "image_url",
        "",
    )

    img_name = os.path.basename(
        image_url
    )

    img_path = UPLOADS / img_name

    if not img_path.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "The analyzed image is no longer "
                "on disk. Re-run the analysis first."
            ),
        )

    # --------------------------------------------------------
    # Media type
    # --------------------------------------------------------

    media = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(
        img_path.suffix.lower()
    )

    if not media:

        raise HTTPException(
            status_code=400,
            detail=(
                "PPE vision check supports "
                "JPG, PNG, and WEBP frames."
            ),
        )

    # --------------------------------------------------------
    # Claude PPE Vision
    # --------------------------------------------------------

    try:

        ppe = claude_service.analyze_ppe(
            img_path.read_bytes(),
            media,
            analysis["detections"],
        )

    except RuntimeError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "PPE vision check failed. "
                "Check the backend logs."
            ),
        )

    # --------------------------------------------------------
    # Update worker boxes
    # --------------------------------------------------------

    worker_boxes = [
        box
        for box in analysis["detections"].get(
            "boxes",
            [],
        )
        if box.get("kind")
        in (
            "worker",
            "violation",
            "unclear",
        )
    ]

    for worker in ppe.get(
        "per_worker",
        [],
    ):

        index = worker.get("index")

        if not isinstance(index, int):
            continue

        if index < 0 or index >= len(
            worker_boxes
        ):
            continue

        box = worker_boxes[index]

        missing = [
            item
            for item in (
                "helmet",
                "boots",
            )
            if worker.get(item) == "missing"
        ]

        unclear = [
            item
            for item in (
                "helmet",
                "boots",
            )
            if worker.get(item) == "unclear"
        ]

        # PPE violation
        if missing:

            box["kind"] = "violation"

            box["label"] = (
                "WORKER · ⚠ NO "
                + "+".join(
                    item.upper()
                    for item in missing
                )
            )

        # PPE unclear
        elif unclear:

            box["kind"] = "unclear"

            box["label"] = (
                "WORKER · PPE UNCLEAR ("
                + "/".join(unclear)
                + ")"
            )

        # PPE okay
        else:

            box["kind"] = "worker"

            box["label"] = (
                "WORKER · PPE OK"
            )

    # --------------------------------------------------------
    # Merge PPE result
    # --------------------------------------------------------

    analysis["detections"]["ppe"] = {
        "evaluated": True,

        "method": ppe.get(
            "source",
            "Claude Vision",
        ),

        "workers_checked": ppe.get(
            "workers_seen",
            0,
        ),

        "helmet_violations": (
            ppe.get("helmet", {})
            .get("missing", 0)
        ),

        "helmets_worn": (
            ppe.get("helmet", {})
            .get("wearing", 0)
        ),

        "helmet_unclear": (
            ppe.get("helmet", {})
            .get("unclear", 0)
        ),

        "boot_violations": (
            ppe.get("boots", {})
            .get("missing", 0)
        ),

        "boots_worn": (
            ppe.get("boots", {})
            .get("wearing", 0)
        ),

        "boots_unclear": (
            ppe.get("boots", {})
            .get("unclear", 0)
        ),

        "vest_violations": 0,

        "notes": ppe.get(
            "notes",
            [],
        ),

        "per_worker": ppe.get(
            "per_worker",
            [],
        ),
    }

    # --------------------------------------------------------
    # Re-run AI reasoning
    # --------------------------------------------------------

    try:

        analysis["analysis"] = (
            claude_service.analyze_site(
                analysis["detections"]
            )
        )

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to re-run AI reasoning "
                "after PPE check."
            ),
        )

    # --------------------------------------------------------
    # Update label
    # --------------------------------------------------------

    analysis["data_label"] = (
        analysis["data_label"].split(
            " + "
        )[0]
        + " + AI Vision PPE Check "
        + "+ AI Interpretation"
    )

    # --------------------------------------------------------
    # Save state
    # --------------------------------------------------------

    STATE["analysis"] = analysis

    STATE["alert_history"] = (
        analysis["analysis"].get(
            "alerts"
        )
        or []
    ) + STATE["alert_history"]

    return {
        "ppe": ppe,
        "bundle": analysis,
    }


# ============================================================
# AI REPORT
# ============================================================

@app.post("/api/report")
def generate_report():
    """
    Generate an AI report from the current analysis.
    """

    analysis = _current_analysis()

    try:

        report = (
            claude_service.generate_report(
                analysis
            )
        )

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Report generation failed. "
                "Check the backend logs."
            ),
        )

    return {
        "report": report,
        "data_label": analysis[
            "data_label"
        ],
    }


# ============================================================
# AI ASSISTANT
# ============================================================

@app.post("/api/assistant")
def assistant(
    query: AssistantQuery,
):
    """
    Answer a user question using the current
    SiteGuard analysis.
    """

    if not query.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Please type a question first.",
        )

    analysis = _current_analysis()

    try:

        answer = (
            claude_service.assistant_answer(
                query.question.strip(),
                analysis,
            )
        )

    except Exception:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "AI assistant failed. "
                "Check the backend logs."
            ),
        )

    return {
        "answer": answer
    }


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def unhandled_exception(
    request,
    exc,
):
    """
    Friendly error for frontend.
    Full traceback is printed in server logs.
    """

    traceback.print_exception(
        type(exc),
        exc,
        exc.__traceback__,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "Something went wrong on the server. "
                "Check the backend logs."
            )
        },
    )


# ============================================================
# FRONTEND STATIC FILES
# ============================================================

# This is useful when running the complete project locally.
#
# On Render, your frontend is already hosted on Netlify,
# but keeping this mount does not hurt as long as the
# frontend directory exists in the GitHub repository.

if FRONTEND.exists():

    app.mount(
        "/",
        StaticFiles(
            directory=str(FRONTEND),
            html=True,
        ),
        name="frontend",
    )

else:

    print(
        f"[SiteGuard] Frontend directory not found: {FRONTEND}"
    )