# =============================================================================
# llm/client.py — Ollama LLM client (CPU-optimised for qwen3:4b)
# Phase 2: RAG context injected into prompt via rag_context parameter
# Phase 3: run_deep_analysis() implemented here
# =============================================================================

import json
import requests
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL, LLM_HOST, LLM_TIMEOUT
from agent.tools import (search_cve, lookup_mitre, get_attack_details,
                          estimate_severity, suggest_mitigation)


def _call_ollama(messages: list) -> dict:
    """Single call to Ollama /api/chat endpoint."""
    payload = {
        "model":   LLM_MODEL,
        "messages": messages,
        "stream":  False,
        "options": {"num_ctx": 2048, "temperature": 0.3, "num_predict": 1024}
        # "options": {"num_ctx": 4096, "temperature": 0.3, "num_predict": 512}
        # "options": {"num_ctx": 512, "temperature": 0.3, "num_predict": 256} #2048,1024 this is initial value of num_ctx and num_predict respectively
    }
    resp = requests.post(
        f"{LLM_HOST}/api/chat",
        json=payload,
        timeout=LLM_TIMEOUT
    )
    resp.raise_for_status()
    return resp.json()


def _gather_tool_results(alert: dict) -> dict:
    """Call all tools before sending to LLM."""
    attack_type = alert.get("attack_type", "unknown")
    return {
        "attack_details": get_attack_details(attack_type),
        "mitre":          lookup_mitre(attack_type),
        "cve":            search_cve(attack_type),
        "severity":       estimate_severity(
                              alert.get("anomaly_score", 0),
                              alert.get("lgbm_confidence", 0),
                              alert.get("anomaly_threshold", 1)
                          ),
        "mitigation":     suggest_mitigation(attack_type),
    }


def _build_prompt(alert: dict, tool_results: dict, rag_context: str = "") -> tuple:
    """
    Build system + user prompt with tool results and optional RAG context.
    Phase 2: rag_context injected here when RAG_ENABLED = True
    """
    # /no_think disables thinking mode for qwen3
    # system = "/no_think\nYou are an IoT/IoMT cybersecurity analyst. Write a clear structured security report based on the alert and intelligence provided."
    system = "/no_think\nYou are an IoT security analyst. Analyse the alert and write a structured report."

    mitre      = tool_results["mitre"].get("technique", {})
    cves       = tool_results["cve"].get("results", [])
    severity   = tool_results["severity"]
    mitigation = tool_results["mitigation"].get("mitigation_steps", [])
    details    = tool_results["attack_details"].get("details", "Unknown attack.")

    cve_text = "; ".join([
        f"{c.get('cve_id','N/A')}(CVSS:{c.get('cvss_score','N/A')})"
        for c in cves[:2]
    ]) or "None found"

    mit_text = " | ".join(mitigation[:3])

    # RAG context block — empty in Phase 1, populated in Phase 2
    # rag_block = f"\n{rag_context[:150]}\n" if rag_context else ""
    rag_block = f"\n{rag_context[:50]}\n" if rag_context else ""

#     user = f"""ALERT:
# Attack : {alert['attack_type']}
# LightGBM: {alert['lgbm_label']} ({alert['lgbm_confidence']*100:.0f}%)
# Anomaly : {alert['is_anomaly']} (score:{alert['anomaly_score']})
# Severity: {alert['severity']}
# {rag_block}
# INTELLIGENCE:
# Details : {details[:200]}
# MITRE   : {mitre.get('id','N/A')} - {mitre.get('name','N/A')} ({mitre.get('tactic','N/A')})
# CVEs    : {cve_text}
# Severity: {severity.get('severity','N/A')} CVSS:{severity.get('cvss_estimate','N/A')}
# Mitigation: {mit_text}

# Write a structured security report:
# 1) Attack Summary 2) MITRE Mapping 3) CVEs 4) Severity 5) Mitigation Steps"""
    user = f"""ALERT:
 Attack : {alert['attack_type']} | Severity: {alert['severity']}
 LightGBM: {alert['lgbm_label']} ({alert['lgbm_confidence']*100:.0f}%)
 Anomaly : {alert['is_anomaly']} (score:{alert['anomaly_score']})
 {rag_block}
 INTELLIGENCE:
 Details : {details[:200]}
 MITRE   : {mitre.get('id','N/A')} - {mitre.get('name','N/A')} ({mitre.get('tactic','N/A')})
 CVEs    : {cve_text}
 Severity: {severity.get('severity','N/A')} CVSS:{severity.get('cvss_estimate','N/A')}
 Mitigation: {mit_text[:100]}

 Write a security report with these sections:
 1) Attack Summary
 2) MITRE Mapping
 3) CVEs
 4) Severity
 5) Mitigation Steps"""

    return system, user


def run_superficial_analysis(system_prompt: str, user_prompt: str,
                              alert: dict = None, rag_context: str = "") -> dict:
    """
    Run LLM superficial analysis.
    Tools pre-called, results + RAG context injected into prompt.
    Phase 2: rag_context passed from orchestrator._retrieve_rag_context()
    """
    print("[LLM] Gathering tool results...")
    tool_results = _gather_tool_results(alert or {})

    tool_calls_made = [
        {"tool": "get_attack_details", "result": tool_results["attack_details"]},
        {"tool": "lookup_mitre",       "result": tool_results["mitre"]},
        {"tool": "search_cve",         "result": tool_results["cve"]},
        {"tool": "estimate_severity",  "result": tool_results["severity"]},
        {"tool": "suggest_mitigation", "result": tool_results["mitigation"]},
    ]

    print("[LLM] Sending to qwen3:4b...")
    system, user = _build_prompt(alert or {}, tool_results, rag_context)
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

    try:
        # response = _call_ollama(messages)
        # raw = response.get("message", {}).get("content", "")
        # print(f"[DEBUG] Raw length: {len(raw)}")
        # print(f"[DEBUG] Raw response: {raw[:300]}")
        # answer = raw
        response = _call_ollama(messages)
        answer   = response.get("message", {}).get("content", "No response.")
        if "</think>" in answer:
            stripped = answer.split("</think>")[-1].strip()
            answer = stripped if stripped else answer.split("</think>")[0].strip()
        status = "ok"
        response = _call_ollama(messages)
        raw = response.get("message", {}).get("content", "")
        # print(f"[DEBUG] Raw LLM response length: {len(raw)}")
        # print(f"[DEBUG] First 200 chars: {raw[:200]}")
        answer = raw
    except Exception as e:
        answer = f"LLM error: {str(e)}"
        status = "error"

    print(f"[LLM] Response received. Status: {status}")

    return {
        "final_answer":    answer,
        "tool_calls_made": tool_calls_made,
        "reasoning_trace": [{"iteration": 1, "content": answer, "tool_calls": []}],
        "status":          status
    }


# ── Phase 3 stub — Deep Analysis ──────────────────────────────────────────────
def run_deep_analysis(system_prompt: str, user_prompt: str) -> dict:
    """Phase 3: Deep log file analysis — not yet implemented."""
    return {
        "final_answer":    "Deep analysis pending (Phase 3).",
        "tool_calls_made": [],
        "reasoning_trace": [],
        "status":          "not_implemented"
    }
