"""Claude service — the reasoning and decision-support layer.

Computer vision detects. Claude understands, explains, prioritizes, reports.
Claude is always grounded in the structured detection JSON it is given and is
instructed never to invent events. If no API key is configured (or the call
fails mid-demo), a deterministic rule-based fallback keeps the demo alive and
is labeled as such.
"""
from __future__ import annotations

import json
import os

MODEL = "claude-sonnet-4-6"
_client = None


def is_available() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def _ask(system: str, user: str, max_tokens: int = 1800) -> str:
    client = _get_client()
    msg = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


GROUNDING = (
    "You are SiteGuard AI's reasoning layer for a construction-site intelligence "
    "platform. You receive structured detection data produced by a computer-vision "
    "layer (and, in demo mode, clearly labeled prototype demo data).\n"
    "Hard rules:\n"
    "1. NEVER invent detections, events, people, equipment, or numbers not present in the data.\n"
    "2. Clearly separate: Detected facts vs AI interpretation vs Recommendations.\n"
    "3. If PPE or a zone was marked evaluated=false, say it was not evaluated — do not report violations or compliance for it.\n"
    "4. A single image cannot prove long-term idle time; call such items 'potentially idle observations'.\n"
    "5. Be concise, professional, and useful to a site supervisor.\n"
    "6. For PPE findings, use typed alert titles: 'PPE Violation — Missing Helmet', "
    "'PPE Violation — Missing Safety Boots', 'PPE Violation — Missing Hi-Vis Vest', "
    "'PPE Assessment Incomplete' for unclear items. Escalate a missing helmet to critical "
    "when the same frame shows worker-equipment proximity. Use per_worker data when present "
    "to say WHICH workers (by index) are affected."
)


# ---------------------------------------------------------------- safety analysis
def analyze_site(detections: dict) -> dict:
    """Safety analysis + risk prioritization + equipment insight, as structured JSON."""
    system = GROUNDING + (
        "\nRespond with ONLY a JSON object (no markdown fences, no preamble) with keys:\n"
        '{"risk_score": int 0-100 (higher = safer),'
        ' "risk_level": "Critical|High|Medium|Low",'
        ' "summary": str (2-3 sentences),'
        ' "alerts": [{"severity":"critical|high|medium|info","title":str,'
        '"detected":str,"interpretation":str,"recommendation":str}],'
        ' "equipment_insights": [{"type":str,"observation":str,"suggestion":str}],'
        ' "recommendations": [str, ...] (top 3-5 actions in priority order)}'
    )
    user = "Structured site detection data:\n" + json.dumps(detections, indent=2)
    try:
        raw = _ask(system, user)
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        parsed["engine"] = "claude"
        return parsed
    except Exception:
        return _fallback_analysis(detections)


def _fallback_analysis(d: dict) -> dict:
    """Deterministic offline fallback so the live demo never dies. Labeled as fallback."""
    alerts = []
    score = 90
    ppe = d.get("ppe") or {}
    near_equipment = bool(d.get("proximity_observations"))
    if ppe.get("evaluated"):
        hv, vv = ppe.get("helmet_violations", 0), ppe.get("vest_violations", 0)
        bv = ppe.get("boot_violations", 0)
        if bv:
            score -= 6 * bv
            alerts.append({
                "severity": "high", "title": f"PPE Violation — Missing Safety Boots ({bv} worker" + ("s" if bv > 1 else "") + ")",
                "detected": f"{bv} worker(s) assessed without safety footwear.",
                "interpretation": "Puncture, crush, and slip injuries to the foot are common where debris, rebar, and vehicles are present.",
                "recommendation": "Direct the affected worker(s) to the site office for safety boots before continuing work.",
            })
        if hv:
            score -= 10 * hv
            alerts.append({
                "severity": "critical" if near_equipment else "high",
                "title": f"PPE Violation — Missing Helmet ({hv} worker" + ("s" if hv > 1 else "") + ")",
                "detected": f"{hv} worker(s) assessed without head protection." + (" Worker-equipment proximity was also observed in this frame." if near_equipment else ""),
                "interpretation": "Head injury is a leading cause of construction fatalities; risk escalates near operating machinery." if near_equipment else "Missing head protection raises struck-by and falling-object injury risk.",
                "recommendation": "Stop the affected worker(s), issue helmets before they re-enter the work area, and log the violation.",
            })
        if vv:
            score -= 6 * vv
            alerts.append({
                "severity": "medium", "title": f"PPE Violation — Missing Hi-Vis Vest ({vv} worker{'s' if vv > 1 else ''})",
                "detected": f"{vv} worker(s) observed without a high-visibility vest.",
                "interpretation": "Low visibility increases struck-by risk around vehicles.",
                "recommendation": "Issue vests and re-brief visibility requirements.",
            })
    unclear_total = (ppe.get("helmet_unclear", 0) or 0) + (ppe.get("boots_unclear", 0) or 0)
    if ppe.get("evaluated") and unclear_total:
        alerts.append({
            "severity": "info", "title": f"PPE Assessment Incomplete ({unclear_total} item{'s' if unclear_total > 1 else ''} unclear)",
            "detected": "Some workers' heads or feet were occluded or cropped in the frame.",
            "interpretation": "Camera angle or obstructions prevented a confident PPE judgment for these items.",
            "recommendation": "Spot-check these workers in person or capture a clearer angle.",
        })
    rz = d.get("restricted_zone_events")
    if isinstance(rz, dict) and rz.get("evaluated") and rz.get("count"):
        score -= 8
        alerts.append({
            "severity": "medium", "title": "Restricted-zone activity",
            "detected": f"{rz['count']} restricted-zone event(s) observed.",
            "interpretation": "Entry into a marked zone may indicate a control breakdown.",
            "recommendation": "Check zone barriers and signage; confirm authorization.",
        })
    for p in d.get("proximity_observations", []):
        score -= 8
        alerts.append({
            "severity": "critical", "title": "Worker near heavy equipment",
            "detected": f"A worker was observed close to {p['worker_near']} (single frame).",
            "interpretation": "Close worker-equipment proximity is a leading struck-by risk factor.",
            "recommendation": "Confirm spotter coverage and exclusion distance at this location.",
        })
    if not alerts:
        alerts.append({
            "severity": "info", "title": "No safety observations in this frame",
            "detected": f"{d.get('workers', 0)} worker(s) and {d.get('equipment_count', 0)} equipment unit(s) detected.",
            "interpretation": "No risk indicators surfaced from the available data.",
            "recommendation": "Continue routine monitoring.",
        })
    idle = [e for e in d.get("equipment_observations", []) if "idle" in str(e.get("status", ""))]
    return {
        "engine": "fallback (Claude API unavailable — rule-based prototype logic)",
        "risk_score": max(20, min(score, 98)),
        "risk_level": "High" if score < 70 else ("Medium" if score < 85 else "Low"),
        "summary": f"{d.get('workers', 0)} workers and {d.get('equipment_count', 0)} equipment units observed. "
                   f"{len(alerts)} finding(s) surfaced from the available data.",
        "alerts": alerts,
        "equipment_insights": [
            {"type": e["type"], "observation": f"Status: {e.get('status', 'observed')}",
             "suggestion": "Review dispatch schedule for this unit." if e in idle else "No action indicated from this data."}
            for e in d.get("equipment_observations", [])
        ],
        "recommendations": [a["recommendation"] for a in alerts][:5],
    }


# ---------------------------------------------------------------- report
def generate_report(analysis_bundle: dict) -> str:
    system = GROUNDING + (
        "\nGenerate a professional markdown report titled 'SiteGuard AI — Site Intelligence Report' "
        "with sections: 1 Executive Summary, 2 Worker Safety, 3 PPE Compliance, 4 Equipment Observations, "
        "5 Detected Risks, 6 Priority Incidents, 7 Recommended Actions, 8 Overall Risk Assessment. "
        "State the data source label verbatim near the top. Keep it under 450 words."
    )
    user = json.dumps({
        "data_label": analysis_bundle["data_label"],
        "timestamp": analysis_bundle["timestamp"],
        "detections": analysis_bundle["detections"],
        "analysis": {k: v for k, v in analysis_bundle["analysis"].items() if k != "engine"},
    }, indent=2)
    try:
        return _ask(system, user, max_tokens=2000)
    except Exception:
        a = analysis_bundle["analysis"]
        lines = [
            "# SiteGuard AI — Site Intelligence Report",
            f"*Data source: {analysis_bundle['data_label']} · {analysis_bundle['timestamp']}*",
            "*(Generated by offline fallback — Claude API unavailable.)*",
            "\n## 1. Executive Summary", a.get("summary", "—"),
            "\n## 2–6. Findings",
        ]
        for al in a.get("alerts", []):
            lines.append(f"- **[{al['severity'].upper()}] {al['title']}** — {al['detected']} {al['interpretation']}")
        lines.append("\n## 7. Recommended Actions")
        lines += [f"{i+1}. {r}" for i, r in enumerate(a.get("recommendations", []))]
        lines.append(f"\n## 8. Overall Risk Assessment\nRisk score {a.get('risk_score','—')}/100 ({a.get('risk_level','—')}).")
        return "\n".join(lines)


# ---------------------------------------------------------------- assistant
def assistant_answer(question: str, analysis_bundle: dict) -> str:
    system = GROUNDING + (
        "\nYou are the dashboard's AI assistant answering a site manager. Answer ONLY from the "
        "provided site-analysis data. If the data cannot answer the question, say exactly that "
        "and suggest what data would be needed. 2-6 sentences unless a report/list is requested."
    )
    user = (
        "Current site analysis data:\n"
        + json.dumps({
            "data_label": analysis_bundle["data_label"],
            "detections": analysis_bundle["detections"],
            "analysis": {k: v for k, v in analysis_bundle["analysis"].items() if k != "engine"},
        }, indent=2)
        + f"\n\nManager question: {question}"
    )
    try:
        return _ask(system, user, max_tokens=900)
    except Exception:
        a = analysis_bundle["analysis"]
        top = a.get("alerts", [{}])[0]
        return (
            "(Offline fallback — Claude API unavailable.) Based on the current analysis: "
            f"{a.get('summary','no summary available')} Highest-priority item: "
            f"{top.get('title','none')} — {top.get('recommendation','continue monitoring')}"
        )


# ---------------------------------------------------------------- PPE vision check
def analyze_ppe(image_bytes: bytes, media_type: str, detections: dict) -> dict:
    """Per-worker helmet & boot assessment via Claude vision.

    We send Claude the image PLUS the numbered worker boxes YOLO found, and it
    judges each numbered worker individually. That lets the UI turn exactly the
    non-compliant workers' rectangles red. Occluded heads/feet must be marked
    'unclear' — never guessed. Counts are computed server-side from the
    per-worker array so arithmetic can't drift.
    """
    if not is_available():
        raise RuntimeError(
            "The PPE vision check needs the Claude API key. Create a file named exactly '.env' in the "
            "project root (the folder that contains backend/ and frontend/) with one line: "
            "ANTHROPIC_API_KEY=sk-ant-your-real-key — no quotes, no spaces — then restart the server. "
            "The server terminal prints 'Claude API key: LOADED' when it worked."
        )
    import base64
    client = _get_client()

    worker_boxes = [b for b in detections.get("boxes", []) if b.get("kind") in ("worker", "violation")]
    box_lines = "\n".join(
        f"Worker {i}: box x1={b['box'][0]}, y1={b['box'][1]}, x2={b['box'][2]}, y2={b['box'][3]} "
        "(normalized 0-1, origin top-left)"
        for i, b in enumerate(worker_boxes)
    ) or "No worker boxes provided — assess any people you can see and index them 0..N in reading order."

    system = (
        "You are SiteGuard AI's PPE vision checker for construction sites. You get a site image and "
        "a numbered list of person bounding boxes from an object detector. For EACH numbered worker, "
        "judge from the image: helmet/hard-hat worn, and safety boots worn. If that body part is "
        "occluded, cropped, or too small to judge, answer 'unclear' — NEVER guess. "
        "Respond with ONLY a JSON object, no markdown fences:\n"
        '{"workers": [{"index": int, "helmet": "worn"|"missing"|"unclear", '
        '"boots": "worn"|"missing"|"unclear", "note": str-or-null}],'
        ' "notes": [str, ...] (0-3 short scene-level observations)}\n'
        "Every provided index MUST appear exactly once in workers."
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=900,
        system=system,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                             "data": base64.b64encode(image_bytes).decode()}},
                {"type": "text", "text": "Detector worker boxes:\n" + box_lines +
                 "\nAssess helmet and boot compliance for each numbered worker."},
            ],
        }],
    )
    raw = "".join(b.text for b in msg.content if b.type == "text").strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    parsed = json.loads(raw)
    per_worker = parsed.get("workers", [])

    def tally(key):
        t = {"wearing": 0, "missing": 0, "unclear": 0}
        for wkr in per_worker:
            v = wkr.get(key, "unclear")
            t["wearing" if v == "worn" else ("missing" if v == "missing" else "unclear")] += 1
        return t

    return {
        "workers_seen": len(per_worker),
        "per_worker": per_worker,
        "helmet": tally("helmet"),
        "boots": tally("boots"),
        "notes": parsed.get("notes", []),
        "source": "AI Vision Observation (Claude) — approximate, verify on site",
    }
