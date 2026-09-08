"""Isolated YOLO worker — runs in its OWN process so that a torch/DLL crash or
out-of-memory kill can never take down the FastAPI server.

Models:
  models/yolov8n.pt  - base COCO model (workers + vehicles), auto-downloaded
  models/ppe.pt      - OPTIONAL fine-tuned PPE model (run get_ppe_model.py).
                       When present, per-worker helmet/vest/boots compliance is
                       computed for whatever classes that model actually has.

Modes:
  python yolo_worker.py <image_path>
  python yolo_worker.py --video <video> <keyframe_out>
  python yolo_worker.py --serve            # persistent live mode (b64 frames)
"""
import json
import os
import sys
from pathlib import Path

PERSON = {"person"}
CONF_THRESHOLD = 0.35
VIDEO_MAX_FRAMES = 8

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS_DIR))
PPE_WEIGHTS = MODELS_DIR / "ppe.pt"

# Keyword mapping so ANY community PPE model works — we only ever report the
# attributes whose classes actually exist in the loaded model (honesty rule).
HELMET_OK = {"hardhat", "helmet", "hard hat", "hard-hat"}
HELMET_NO = {"no-hardhat", "no_hardhat", "no-helmet", "no_helmet", "nohelmet", "head"}
VEST_OK = {"safety vest", "safety-vest", "vest", "safetyvest"}
VEST_NO = {"no-safety vest", "no-safety-vest", "no_vest", "no-vest", "novest"}
BOOTS_OK = {"boots", "boot", "shoes", "shoe", "footwear", "safety boots"}
BOOTS_NO = {"no-boots", "no-boot", "no_shoes", "no-shoes"}
EQUIPMENT = {"machinery", "vehicle", "truck", "car", "bus", "train", "excavator", "crane", "loader", "dump truck"}
IGNORED = {"safety cone", "mask", "no-mask", "people", "glasses", "gloves", "dog", "cat"}


def classify_detection(name: str, confidence: float):
    """Return a structured detection for workers vs. all other observable objects.

    The stock COCO model does not only return people and vehicles; it can also
    recognize common site objects such as traffic lights, fire hydrants, benches,
    and signs. Those were previously dropped by the hard-coded allowlist, which is
    why the app looked like it only detected workers.
    """
    normalized = (name or "").strip()
    if not normalized:
        return None

    if normalized.lower() in PERSON:
        return {"kind": "worker", "label": "WORKER", "type": "worker"}

    return {
        "kind": "equipment",
        "label": normalized.upper(),
        "type": normalized.lower().replace("_", " "),
    }


def _load_models():
    """Load PPE model if available, else the base COCO model. Returns (model, is_ppe, caps)."""
    os.chdir(MODELS_DIR)
    from ultralytics import YOLO
    if PPE_WEIGHTS.exists():
        model = YOLO(str(PPE_WEIGHTS))
        names = {str(v).lower() for v in model.names.values()}
        caps = {
            "helmet": bool(names & (HELMET_OK | HELMET_NO)),
            "vest": bool(names & (VEST_OK | VEST_NO)),
            "boots": bool(names & (BOOTS_OK | BOOTS_NO)),
        }
        return model, True, caps
    model = YOLO(str(MODELS_DIR / "yolov8n.pt") if (MODELS_DIR / "yolov8n.pt").exists() else "yolov8n.pt")
    return model, False, {"helmet": False, "vest": False, "boots": False}


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _inside(cx, cy, p, y_from=0.0, y_to=1.0, pad=0.02):
    x1, y1, x2, y2 = p
    h = y2 - y1
    return (x1 - pad <= cx <= x2 + pad) and (y1 + y_from * h - pad <= cy <= y1 + y_to * h + pad)


def _detect_source(model, source, is_ppe=False, caps=None):
    results = model(source, verbose=False)[0]
    h, w = results.orig_shape[0], results.orig_shape[1]

    persons, items, boxes, equipment = [], [], [], []
    for b in results.boxes:
        conf = float(b.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        name = str(results.names[int(b.cls[0])]).strip()
        lname = name.lower()
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        nbox = [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4)]

        if lname in PERSON:
            persons.append({"box": nbox, "confidence": round(conf, 2),
                            "helmet": None, "vest": None, "boots": None})
        elif is_ppe and lname in (HELMET_OK | HELMET_NO | VEST_OK | VEST_NO | BOOTS_OK | BOOTS_NO):
            items.append({"name": lname, "box": nbox})
        elif lname in IGNORED:
            continue
        else:
            classification = classify_detection(name, conf)
            if classification is None or classification["kind"] != "equipment":
                continue
            if is_ppe and lname not in EQUIPMENT:
                continue  # unknown PPE-model class: don't misreport it as equipment
            equipment.append({"type": classification["type"], "status": "observed", "confidence": round(conf, 2)})
            boxes.append({"label": classification["label"], "kind": "equipment",
                          "confidence": round(conf, 2), "box": nbox})

    # Associate PPE items to the person whose region they fall in.
    for it in items:
        cx, cy = _center(it["box"])
        n = it["name"]
        for p in persons:
            if n in (HELMET_OK | HELMET_NO) and _inside(cx, cy, p["box"], 0.0, 0.55):
                p["helmet"] = (n in HELMET_OK) if p["helmet"] is not False else False
                break
            if n in (VEST_OK | VEST_NO) and _inside(cx, cy, p["box"], 0.15, 0.75):
                p["vest"] = (n in VEST_OK) if p["vest"] is not False else False
                break
            if n in (BOOTS_OK | BOOTS_NO) and _inside(cx, cy, p["box"], 0.65, 1.05):
                p["boots"] = (n in BOOTS_OK) if p["boots"] is not False else False
                break

    caps = caps or {}
    ppe = None
    if is_ppe:
        def tally(attr):
            ok = sum(1 for p in persons if p[attr] is True)
            no = sum(1 for p in persons if p[attr] is False)
            unk = len(persons) - ok - no
            return {"evaluated": True, "ok": ok, "violations": no, "not_visible": unk}
        ppe = {
            "evaluated": True,
            "model": "ppe.pt (fine-tuned)",
            "workers_checked": len(persons),
            "helmet": tally("helmet") if caps.get("helmet") else {"evaluated": False, "note": "loaded PPE model has no helmet class"},
            "vest": tally("vest") if caps.get("vest") else {"evaluated": False, "note": "loaded PPE model has no vest class"},
            "boots": tally("boots") if caps.get("boots") else {"evaluated": False, "note": "loaded PPE model has no boots class — see get_ppe_model.py for options"},
        }
        # legacy flat keys used elsewhere
        ppe["helmet_violations"] = ppe["helmet"].get("violations", 0)
        ppe["vest_violations"] = ppe["vest"].get("violations", 0)
        ppe["boot_violations"] = ppe["boots"].get("violations", 0)

    for p in persons:
        flags = []
        if p["helmet"] is False: flags.append("⚠ HELMET")
        if p["vest"] is False: flags.append("⚠ VEST")
        if p["boots"] is False: flags.append("⚠ BOOTS")
        if flags:
            label, kind = "WORKER · " + " ".join(flags), "violation"
        elif is_ppe and (p["helmet"] or p["vest"] or p["boots"]):
            label, kind = "WORKER · PPE OK", "worker"
        else:
            label, kind = "WORKER", "worker"
        boxes.append({"label": label, "kind": kind, "confidence": p["confidence"], "box": p["box"]})

    out = {"workers": len(persons), "equipment": equipment, "boxes": boxes}
    if ppe:
        out["ppe"] = ppe
    return out


def run_image(image_path: str) -> int:
    model, is_ppe, caps = _load_models()
    print(json.dumps(_detect_source(model, image_path, is_ppe, caps)))
    return 0


def run_video(video_path: str, keyframe_out: str) -> int:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("could not open video", file=sys.stderr)
        return 3
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    duration = round(total / fps, 1) if fps else 0
    n = min(VIDEO_MAX_FRAMES, total) if total else VIDEO_MAX_FRAMES
    indices = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)] if total else list(range(n))

    model, is_ppe, caps = _load_models()
    frames, raw_frames, key_i, key_score = [], [], 0, -1
    for fi, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        det = _detect_source(model, frame, is_ppe, caps)
        det["t"] = round(idx / fps, 1) if fps else fi
        frames.append(det)
        raw_frames.append(frame)
        score = det["workers"] + len(det["equipment"])
        if score > key_score:
            key_score, key_i = score, len(frames) - 1
    cap.release()
    if not frames:
        print("no readable frames in video", file=sys.stderr)
        return 3
    cv2.imwrite(keyframe_out, raw_frames[key_i])
    print(json.dumps({
        "video": {"duration_s": duration, "fps": round(fps, 1), "frames_sampled": len(frames)},
        "frames": frames,
        "key_frame_index": key_i,
    }))
    return 0


def run_serve() -> int:
    """Persistent live mode: b64-jpeg lines in, JSON lines out. In-memory only."""
    model, is_ppe, caps = _load_models()
    print(json.dumps({"ready": True, "ppe": is_ppe}), flush=True)
    import base64
    import numpy as np
    import cv2
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            if line.startswith("b64:"):
                buf = np.frombuffer(base64.b64decode(line[4:]), dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("could not decode frame")
                print(json.dumps(_detect_source(model, frame, is_ppe, caps)), flush=True)
            else:
                print(json.dumps(_detect_source(model, line, is_ppe, caps)), flush=True)
        except Exception as e:
            print(json.dumps({"error": str(e)[:200]}), flush=True)
    return 0


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--serve":
        return run_serve()
    if len(sys.argv) >= 4 and sys.argv[1] == "--video":
        return run_video(sys.argv[2], sys.argv[3])
    if len(sys.argv) >= 2:
        return run_image(sys.argv[1])
    print("usage: yolo_worker.py <image> | --video <video> <keyframe_out> | --serve", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
