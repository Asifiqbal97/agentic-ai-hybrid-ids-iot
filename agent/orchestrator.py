# =============================================================================
# agent/orchestrator.py — ML Agent with selective LLM invocation
# Selective policy:
#   - Zero-day (unknown_anomaly) → always invoke LLM
#   - Known attack CVSS >= 7.0 (HIGH/CRITICAL) → invoke LLM
#   - Known attack CVSS < 7.0 (LOW/MEDIUM) → templated report, no LLM
#   - Campaign deduplication: 60s window, only first alert triggers LLM
# Phase 3: enable deep analysis via DEEP_ANALYSIS_ENABLED in config.py
# =============================================================================

import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (SUPERFICIAL_ANALYSIS_ENABLED, DEEP_ANALYSIS_ENABLED,
                    RAG_ENABLED)
from agent.prompts import build_superficial_prompt
from agent.campaign import get_tracker
from agent.tools import (get_attack_details, lookup_mitre,
                          estimate_severity, suggest_mitigation, search_cve)
from llm.client import run_superficial_analysis, run_deep_analysis

# CVSS threshold for LLM invocation
LLM_CVSS_THRESHOLD = 7.0


def _should_invoke_llm(alert: dict) -> bool:
    """
    Selective invocation policy:
    - Zero-day → always True
    - CVSS >= 7.0 (HIGH/CRITICAL) → True
    - CVSS < 7.0 (LOW/MEDIUM known attack) → False
    """
    # Zero-day always gets LLM
    if alert.get("is_anomaly") and not alert.get("is_known_attack"):
        return True
    if alert.get("attack_type") == "unknown_anomaly":
        return True

    # Check severity
    severity = alert.get("severity", "LOW")
    if severity in ["CRITICAL", "HIGH"]:
        return True

    # Estimate CVSS from tool
    try:
        sev_result = estimate_severity(
            alert.get("anomaly_score", 0),
            alert.get("lgbm_confidence", 0),
            alert.get("anomaly_threshold", 1),
            alert.get("attack_type", "")
        )
        cvss = sev_result.get("cvss_score", 0)
        return float(cvss) >= LLM_CVSS_THRESHOLD
    except Exception:
        return True   # default to invoking LLM if unsure


def _build_templated_result(alert: dict) -> dict:
    """
    Build report from deterministic tool outputs — no LLM call.
    Used for LOW/MEDIUM known attacks that don't meet invocation threshold.
    """
    attack_type  = alert.get("attack_type", "unknown")
    tool_results = {
        "attack_details": get_attack_details(attack_type),
        "mitre":          lookup_mitre(attack_type),
        "cve":            search_cve(attack_type),
        "severity":       estimate_severity(
                              alert.get("anomaly_score", 0),
                              alert.get("lgbm_confidence", 0),
                              alert.get("anomaly_threshold", 1),
                              attack_type
                          ),
        "mitigation":     suggest_mitigation(attack_type),
    }

    mitre    = tool_results["mitre"].get("technique", {})
    severity = tool_results["severity"]
    mit      = tool_results["mitigation"].get("mitigation_steps", [])
    details  = tool_results["attack_details"].get("details", "")
    cves     = tool_results["cve"].get("results", [])
    cve_txt  = "; ".join(f"{c.get('cve_id','N/A')} (CVSS:{c.get('cvss_score','N/A')})"
                          for c in cves[:2]) or "None found"
    mit_txt  = "\n".join(f"{i+1}. {s}" for i,s in enumerate(mit[:4]))

    answer = f"""## IDS Alert Report (Templated — Low Severity)

**Attack Type:** {attack_type}
**Details:** {details}

**MITRE ATT&CK:** {mitre.get('id','N/A')} — {mitre.get('name','N/A')} ({mitre.get('tactic','N/A')})

**CVEs:** {cve_txt}

**Severity:** {severity.get('severity','N/A')} (CVSS: {severity.get('cvss_score','N/A')})
**CVSS Vector:** {severity.get('cvss_vector','N/A')}

**Mitigation Steps:**
{mit_txt}

*Note: LLM enrichment withheld — severity below threshold (CVSS < {LLM_CVSS_THRESHOLD}). Report generated from deterministic tool outputs.*"""

    return {
        "final_answer":    answer,
        "tool_calls_made": list(tool_results.keys()),
        "reasoning_trace": [],
        "status":          "templated",
        "llm_invoked":     False,
    }


def handle_alert(alert: dict) -> dict:
    """
    Main entry point for the ML Agent.
    Applies selective LLM invocation + campaign deduplication.
    """
    print(f"\n[Agent] Alert: {alert['attack_type']} | {alert['severity']}")

    result = {
        "alert":                alert,
        "superficial_analysis": None,
        "deep_analysis":        None,
        "llm_invoked":          False,
        "campaign_id":          None,
        "is_new_campaign":      True,
    }

    # ── Campaign deduplication ────────────────────────────────────────────────
    tracker         = get_tracker()
    is_new, camp_id = tracker.check(alert)
    result["campaign_id"]     = camp_id
    result["is_new_campaign"] = is_new

    if not is_new:
        print(f"[Agent] Duplicate campaign {camp_id} — logging only, no LLM")
        result["superficial_analysis"] = {
            "final_answer":    f"Duplicate alert — campaign {camp_id} already analysed.",
            "tool_calls_made": [],
            "reasoning_trace": [],
            "status":          "campaign_duplicate",
            "llm_invoked":     False,
        }
        return result

    # ── RAG context ───────────────────────────────────────────────────────────
    rag_context = ""
    if RAG_ENABLED:
        rag_context = _retrieve_rag_context(alert)

    # ── Selective LLM invocation ──────────────────────────────────────────────
    if SUPERFICIAL_ANALYSIS_ENABLED:
        invoke_llm = _should_invoke_llm(alert)

        if invoke_llm:
            print("[Agent] LLM invoked (zero-day or HIGH/CRITICAL)")
            system_prompt, user_prompt = build_superficial_prompt(alert, rag_context)
            llm_result = run_superficial_analysis(
                system_prompt, user_prompt, alert=alert)
            llm_result["llm_invoked"] = True
            result["llm_invoked"]     = True
        else:
            print("[Agent] LLM skipped (LOW/MEDIUM known attack) — templated report")
            llm_result = _build_templated_result(alert)

        result["superficial_analysis"] = llm_result
        print(f"[Agent] Done. Status: {llm_result['status']}")

    # ── Deep Analysis (Phase 3) ───────────────────────────────────────────────
    if DEEP_ANALYSIS_ENABLED:
        result["deep_analysis"] = run_deep_analysis(
            system_prompt="", user_prompt="")

    return result


def _retrieve_rag_context(alert: dict) -> str:
    """Phase 2: RAG retrieval."""
    try:
        from rag.retriever import retrieve
        query   = f"{alert.get('attack_type','')} IoT attack mitigation"
        context = retrieve(query, top_k=3)
        print(f"[Agent] RAG retrieved ({len(context)} chars)")
        return context
    except Exception as e:
        print(f"[Agent] RAG error: {e}")
        return ""
