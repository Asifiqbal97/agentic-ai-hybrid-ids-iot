# =============================================================================
# reports/writer.py — Format and save analysis reports
# Produces: JSON (machine-readable) + Markdown (human-readable)
# Phase 3: deep_analysis section auto-populated when available
# =============================================================================

import os
import json
import datetime
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORT_DIR


def write_report(agent_result: dict) -> str:
    """
    Generate and save JSON + Markdown reports.
    Returns path to the Markdown report.
    """
    alert    = agent_result["alert"]
    sa       = agent_result.get("superficial_analysis") or {}
    da       = agent_result.get("deep_analysis")        # Phase 3

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base_name = f"report_{timestamp}_{alert['attack_type'].replace(' ', '_')}"

    json_path = os.path.join(REPORT_DIR, base_name + ".json")
    md_path   = os.path.join(REPORT_DIR, base_name + ".md")

    # ── JSON report ───────────────────────────────────────────────────────────
    report_json = {
        "generated_at":        datetime.datetime.utcnow().isoformat(),
        "alert":               alert,
        "superficial_analysis": sa,
        "deep_analysis":        da or "pending",   # Phase 3 fills this
        "tool_calls":          sa.get("tool_calls_made", []),
    }
    with open(json_path, "w") as f:
        json.dump(report_json, f, indent=2)

    # ── Markdown report ───────────────────────────────────────────────────────
    tools_used = ", ".join([t["tool"] for t in sa.get("tool_calls_made", [])]) or "none"
    md = f"""# IDS Alert Report

**Generated:** {datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")} UTC

---

## Alert Summary

| Field             | Value |
|-------------------|-------|
| Attack Type       | `{alert['attack_type']}` |
| Known Attack      | {alert['is_known_attack']} |
| Zero-Day Anomaly  | {alert['is_anomaly']} |
| Severity          | **{alert['severity']}** |
| LightGBM Label    | {alert['lgbm_label']} ({alert['lgbm_confidence']*100:.1f}% confidence) |
| Anomaly Score     | {alert['anomaly_score']} (threshold: {alert['anomaly_threshold']}) |
| Source IP         | {alert['src_ip']} |
| Destination IP    | {alert['dst_ip']} |
| Protocol          | {alert['protocol']} |
| Timestamp         | {alert['timestamp']} |

---

## Top Contributing Features

{_format_features(alert.get('top_features', {}))}

---


if alert.get("zeroday_mitigation"):
    md += "\n## ⚠️ Zero-Day Mitigation (Pre-LLM)\n\n"
    md += alert["zeroday_mitigation"] + "\n\n"
    md += "---\n\n"

## LLM Analysis (Superficial)

**Tools used:** {tools_used}

{sa.get('final_answer', 'No analysis available.')}

---

## Deep Analysis

{_format_deep(da)}

---

## Reasoning Trace

{_format_trace(sa.get('reasoning_trace', []))}

---
*Report saved: {json_path}*
"""

    with open(md_path, "w") as f:
        f.write(md)

    print(f"[Report] Saved → {md_path}")
    return md_path


def _format_features(features: dict) -> str:
    if not features:
        return "_No feature data available._"
    rows = "\n".join([f"- `{k}`: {v}" for k, v in features.items()])
    return rows


def _format_deep(da) -> str:
    if da is None:
        return "_Deep analysis pending (Phase 3)._"
    if isinstance(da, dict):
        return da.get("final_answer", "_No deep analysis output._")
    return str(da)


def _format_trace(trace: list) -> str:
    if not trace:
        return "_No reasoning trace available._"
    out = []
    for step in trace:
        tools = ", ".join(step.get("tool_calls", [])) or "none"
        out.append(f"**Step {step['iteration']}** (tools: {tools})\n\n{step.get('content','')}\n")
    return "\n---\n".join(out)
