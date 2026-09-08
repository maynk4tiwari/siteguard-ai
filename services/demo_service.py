"""Demo Mode dataset — clearly labeled Prototype Demo Data.

This simulates what a fully deployed SiteGuard installation (with a fine-tuned
PPE/equipment model and multi-frame tracking) would produce. It is never
presented as live detection accuracy.
"""


def demo_detections() -> dict:
    return {
        "mode": "demo",
        "data_label": "Prototype Demo Data (simulated detections)",
        "workers": 18,
        "equipment_count": 6,
        "equipment_observations": [
            {"type": "excavator", "status": "active", "confidence": 0.91},
            {"type": "excavator", "status": "potentially_idle", "confidence": 0.88,
             "note": "no movement across sampled frames (demo)"},
            {"type": "truck", "status": "active", "confidence": 0.93},
            {"type": "truck", "status": "active", "confidence": 0.90},
            {"type": "truck", "status": "potentially_idle", "confidence": 0.86,
             "note": "stationary at loading bay across sampled frames (demo)"},
            {"type": "crane", "status": "active", "confidence": 0.84},
        ],
        "ppe": {
            "evaluated": True,
            "model": "Prototype Demo Data",
            "workers_checked": 18,
            "helmet": {"evaluated": True, "ok": 16, "violations": 2, "not_visible": 0},
            "vest": {"evaluated": True, "ok": 17, "violations": 1, "not_visible": 0},
            "boots": {"evaluated": True, "ok": 15, "violations": 1, "not_visible": 2},
            "helmet_violations": 2,
            "vest_violations": 1,
            "boot_violations": 1,
        },
        "restricted_zone_events": {"evaluated": True, "count": 1,
                                   "note": "worker entered marked exclusion zone near crane"},
        "proximity_observations": [
            {"worker_near": "EXCAVATOR", "note": "worker within exclusion distance of slewing excavator"}
        ],
        "boxes": [
            {"label": "WORKER · PPE OK", "kind": "worker", "confidence": 0.94, "box": [0.06, 0.55, 0.12, 0.82]},
            {"label": "WORKER · PPE OK", "kind": "worker", "confidence": 0.92, "box": [0.16, 0.58, 0.22, 0.84]},
            {"label": "WORKER · ⚠ HELMET", "kind": "violation", "confidence": 0.90, "box": [0.40, 0.52, 0.46, 0.80]},
            {"label": "WORKER · ⚠ HELMET", "kind": "violation", "confidence": 0.88, "box": [0.63, 0.60, 0.69, 0.86]},
            {"label": "WORKER · ⚠ VEST", "kind": "violation", "confidence": 0.87, "box": [0.74, 0.57, 0.80, 0.83]},
            {"label": "WORKER · ⚠ BOOTS", "kind": "violation", "confidence": 0.85, "box": [0.24, 0.60, 0.30, 0.87]},
            {"label": "EXCAVATOR · ACTIVE", "kind": "equipment", "confidence": 0.91, "box": [0.28, 0.30, 0.52, 0.72]},
            {"label": "EXCAVATOR · IDLE?", "kind": "equipment", "confidence": 0.88, "box": [0.80, 0.32, 0.98, 0.66]},
            {"label": "TRUCK · ACTIVE", "kind": "equipment", "confidence": 0.93, "box": [0.02, 0.28, 0.20, 0.50]},
            {"label": "CRANE · ACTIVE", "kind": "equipment", "confidence": 0.84, "box": [0.55, 0.02, 0.72, 0.55]},
            {"label": "RESTRICTED ZONE · ⚠", "kind": "zone", "confidence": 1.0, "box": [0.52, 0.55, 0.78, 0.95]},
        ],
    }
