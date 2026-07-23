# =============================================================================
# agent/prompts.py — System and alert prompt templates
# Phase 2: inject RAG context into SYSTEM_PROMPT
# Phase 3: extend DEEP_ANALYSIS_PROMPT
# =============================================================================

SYSTEM_PROMPT = SYSTEM_PROMPT = """You are an IoT security analyst. Analyse the IDS alert using your tools and provide:
1. Attack identification
2. MITRE ATT&CK mapping
3. Relevant CVEs
4. Severity
5. Mitigation steps
{rag_context}"""

# Phase 1: Superficial analysis prompt
SUPERFICIAL_PROMPT = """
## IDS Alert

- **Attack Type**: {attack_type}
- **Known Attack**: {is_known_attack}
- **Zero-Day Anomaly**: {is_anomaly}
- **LightGBM Label**: {lgbm_label} (confidence: {lgbm_confidence})
- **Anomaly Score**: {anomaly_score} (threshold: {anomaly_threshold})
- **Severity**: {severity}
- **Source IP**: {src_ip}
- **Destination IP**: {dst_ip}
- **Protocol**: {protocol}
- **Top Features**: {top_features}

Analyse this alert. Use your tools, then provide:
1. Attack identification and explanation
2. MITRE ATT&CK mapping
3. Relevant CVEs
4. Severity assessment
5. Immediate mitigation steps
"""

# Phase 3 placeholder — Deep Analysis prompt (not used in Phase 1)
DEEP_ANALYSIS_PROMPT = """
## Deep Analysis Request

Previous superficial analysis: {superficial_summary}
Log file path: {log_file_path}
Session ID: {session_id}

Perform deep analysis:
1. Analyse log file for repeated attempts
2. Detect low-rate attack patterns
3. Identify attack timeline
4. Provide forensic summary
"""


def build_superficial_prompt(alert: dict, rag_context: str = "") -> tuple[str, str]:
    """Build system + user prompt for superficial analysis."""
    system = SYSTEM_PROMPT.format(rag_context=rag_context)
    user   = SUPERFICIAL_PROMPT.format(
        attack_type      = alert["attack_type"],
        is_known_attack  = alert["is_known_attack"],
        is_anomaly       = alert["is_anomaly"],
        lgbm_label       = alert["lgbm_label"],
        lgbm_confidence  = alert["lgbm_confidence"],
        anomaly_score    = alert["anomaly_score"],
        anomaly_threshold= alert["anomaly_threshold"],
        severity         = alert["severity"],
        src_ip           = alert["src_ip"],
        dst_ip           = alert["dst_ip"],
        protocol         = alert["protocol"],
        top_features     = alert["top_features"],
    )
    return system, user
