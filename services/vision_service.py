"""Vision service — pretrained YOLOv8n (COCO), executed in an ISOLATED SUBPROCESS.

Why a subprocess? On many laptops (especially Windows), the first torch/YOLO run
can hard-crash the Python process (missing DLLs, OOM). In-process, that kills
the whole FastAPI server and the browser sees "can't reach the backend".
In a subprocess, the crash is contained: the server survives and returns a
clear, friendly error instead.

Honesty note (core project rule): stock COCO detects `person` and generic
vehicles. It does NOT detect helmets, vests, excavators, or cranes — those
appear only in clearly labeled Demo Mode. We never claim a detection this
model cannot make.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

WORKER = Path(__file__).resolve().parent / "yolo_worker.py"
TIMEOUT_S = 300  # generous: first run downloads yolov8n.pt (~6 MB)


class VisionUnavailable(RuntimeError):
    pass


def is_available() -> bool:
    return importlib.util.find_spec("ultralytics") is not None


def detect(image_path: str) -> dict:
    if not is_available():
        raise VisionUnavailable("ultralytics is not installed")

    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER), image_path],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Detection timed out. The first run downloads the YOLO model — "
            "check your internet connection, then try again."
        )

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        print("[vision worker failed]\n" + "\n".join(tail), flush=True)  # server terminal only
        low = (proc.stderr or "").lower()
        if "no module named" in low:
            raise VisionUnavailable("ultralytics/torch import failed — reinstall with: pip install ultralytics")
        if "download" in low or "urlopen" in low or "connection" in low:
            raise RuntimeError("Couldn't download the YOLO model weights. Check your internet connection and retry.")
        raise RuntimeError(
            "The detection process crashed (this is isolated — the server is fine). "
            "Common fixes: pip install --upgrade ultralytics torch, or free some RAM. "
            "Details are in the server terminal."
        )

    try:
        raw = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        raise RuntimeError("Detection produced unreadable output. See the server terminal for details.")

    boxes = raw["boxes"]
    ppe = raw.get("ppe") or {"evaluated": False, "note": "PPE model not installed. Run: python get_ppe_model.py (adds real helmet & vest detection)."}
    return {
        "mode": "live",
        "data_label": "Detected Observation (" + ("YOLOv8 fine-tuned PPE model" if raw.get("ppe") else "YOLOv8n pretrained") + ")",
        "workers": raw["workers"],
        "equipment_count": len(raw["equipment"]),
        "equipment_observations": raw["equipment"],
        "ppe": ppe,
        "restricted_zone_events": {"evaluated": False},
        "boxes": boxes,
        "proximity_observations": _proximity(boxes),
    }


def _proximity(boxes: list[dict]) -> list[dict]:
    """Flag workers whose box center is close to an equipment box center.
    Single-frame observation for Claude to interpret — never a verdict."""
    events = []
    workers = [b for b in boxes if b["kind"] == "worker"]
    machines = [b for b in boxes if b["kind"] == "equipment"]
    for wkr in workers:
        wx = (wkr["box"][0] + wkr["box"][2]) / 2
        wy = (wkr["box"][1] + wkr["box"][3]) / 2
        for m in machines:
            mx = (m["box"][0] + m["box"][2]) / 2
            my = (m["box"][1] + m["box"][3]) / 2
            if abs(wx - mx) < 0.12 and abs(wy - my) < 0.15:
                events.append({"worker_near": m["label"], "note": "single-frame proximity observation"})
                break
    return events


# ===================== VIDEO ANALYSIS =====================
def detect_video(video_path: str, keyframe_out: str) -> dict:
    """Sample frames from a video in the isolated worker, then aggregate.

    Video gives evidence a single image can't: equipment that stays in the same
    place across sampled frames is flagged as a 'potentially idle observation'
    — still an observation for follow-up, never a utilization statistic.
    """
    if not is_available():
        raise VisionUnavailable("ultralytics is not installed")
    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER), "--video", video_path, keyframe_out],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Video analysis timed out. Try a shorter clip (under ~2 minutes).")
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        print("[video worker failed]\n" + "\n".join(tail), flush=True)
        if "could not open video" in (proc.stderr or ""):
            raise RuntimeError("Couldn't read that video. Use MP4 (H.264), WEBM, or AVI.")
        raise RuntimeError("Video analysis crashed (isolated — server is fine). Details in the server terminal.")
    raw = json.loads(proc.stdout.strip().splitlines()[-1])

    frames = raw["frames"]
    key = frames[raw["key_frame_index"]]
    max_workers = max(f["workers"] for f in frames)
    idle = _stationary_equipment(frames)
    equipment = []
    seen = set()
    for f in frames:
        for e in f["equipment"]:
            if e["type"] not in seen:
                seen.add(e["type"])
                e = dict(e)
                e["status"] = "potentially_idle" if e["type"] in idle else "active"
                if e["type"] in idle:
                    e["note"] = f"stationary across {len(frames)} sampled frames (video observation)"
                equipment.append(e)

    boxes = key["boxes"]
    ppe = key.get("ppe")
    if ppe:
        ppe = dict(ppe); ppe["note"] = "computed on the busiest sampled frame"
    else:
        ppe = {"evaluated": False, "note": "PPE model not installed. Run: python get_ppe_model.py"}
    return {
        "mode": "live_video",
        "data_label": f"Detected Observation (YOLOv8n · {len(frames)} frames sampled from video)",
        "workers": max_workers,
        "equipment_count": len(equipment),
        "equipment_observations": equipment,
        "ppe": ppe,
        "restricted_zone_events": {"evaluated": False},
        "boxes": boxes,
        "proximity_observations": _proximity(boxes),
        "video_meta": raw["video"],
        "frame_timeline": [{"t": f["t"], "workers": f["workers"], "equipment": len(f["equipment"])} for f in frames],
    }


def _stationary_equipment(frames: list[dict]) -> set:
    """Equipment types whose box center barely moves between first and last
    sampled frames -> potentially idle observation."""
    if len(frames) < 2:
        return set()
    def centers(f):
        out = {}
        for b in f["boxes"]:
            if b["kind"] == "equipment":
                out.setdefault(b["label"].lower().replace("_", " "), []).append(
                    ((b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2))
        return out
    first, last = centers(frames[0]), centers(frames[-1])
    idle = set()
    for t, pts in first.items():
        for (fx, fy) in pts:
            for (lx, ly) in last.get(t, []):
                if abs(fx - lx) < 0.03 and abs(fy - ly) < 0.03:
                    idle.add(t)
    return idle


# ===================== LIVE FRAME ANALYSIS (persistent worker) =====================
import threading

_live_proc = None
_live_lock = threading.Lock()


def _ensure_live_worker():
    global _live_proc
    if _live_proc is not None and _live_proc.poll() is None:
        return _live_proc
    if not is_available():
        raise VisionUnavailable("ultralytics is not installed")
    _live_proc = subprocess.Popen(
        [sys.executable, str(WORKER), "--serve"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, bufsize=1,
    )
    # Wait for the {"ready": true} line, skipping any model-download progress
    # noise the worker may print first.
    for _ in range(50):
        line = _live_proc.stdout.readline()
        if not line:
            break
        try:
            if json.loads(line).get("ready"):
                return _live_proc
        except (ValueError, AttributeError):
            continue
    _live_proc = None
    raise RuntimeError("Live detection worker failed to start. Check the server terminal.")


def detect_frame(jpeg_bytes: bytes) -> dict:
    """Fast path for live monitoring: one already-loaded model, one frame in,
    boxes out. The frame travels through the pipe as base64 — no temp files,
    so Windows antivirus/sync-folder races can't break it. Auto-restarts the
    worker if it died."""
    import base64
    payload = "b64:" + base64.b64encode(jpeg_bytes).decode("ascii")
    with _live_lock:
        for attempt in (1, 2):
            proc = _ensure_live_worker()
            try:
                proc.stdin.write(payload + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if line:
                    raw = json.loads(line)
                    if "error" in raw:
                        raise RuntimeError(raw["error"])
                    return raw
            except (BrokenPipeError, OSError):
                pass
            global _live_proc
            _live_proc = None  # worker died -> restart once
        raise RuntimeError("Live detection worker keeps crashing. Details in the server terminal.")
