# =============================================================================
# agent/tools.py — Tools available to the LLM agent
# Issue 8: search_cve() now uses local CVE DB — no inference-time API calls
# Run rag/cve_sync.py first to build local DB
# =============================================================================

import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CVE_DB_PATH

# CVE local DB cache
_cve_db = None

def _load_cve_db() -> list:
    global _cve_db
    if _cve_db is not None:
        return _cve_db
    if not os.path.exists(CVE_DB_PATH):
        print(f"[Tools] CVE DB not found. Run: python rag/cve_sync.py")
        _cve_db = []
        return _cve_db
    with open(CVE_DB_PATH) as f:
        data = json.load(f)
    _cve_db = data.get("cves", [])
    print(f"[Tools] CVE DB loaded: {len(_cve_db)} entries")
    return _cve_db


def search_cve(keyword: str) -> dict:
    """Search local CVE database — fully offline, no NVD API at inference time."""
    cves    = _load_cve_db()
    keyword = keyword.lower()
    results = []
    for cve in cves:
        desc = cve.get("desc", "").lower()
        kw   = cve.get("keyword", "").lower()
        if keyword in desc or keyword in kw or any(
            part in desc for part in keyword.split("-") if len(part) > 3
        ):
            results.append({
                "cve_id":      cve["cve_id"],
                "cvss_score":  cve.get("cvss", "N/A"),
                "severity":    cve.get("severity", "N/A"),
                "published":   cve.get("published", "N/A"),
                "description": cve.get("desc", "")[:200],
            })
        if len(results) >= 3:
            break
    if not results:
        return {"status": "ok", "results": [{"info": "No matching CVEs in local DB"}], "source": "local_db"}
    return {"status": "ok", "results": results, "source": "local_db"}


MITRE_LOCAL = {
    "ddos":            {"id": "T1498", "name": "Network Denial of Service",  "tactic": "Impact"},
    "dos":             {"id": "T1499", "name": "Endpoint Denial of Service", "tactic": "Impact"},
    "recon":           {"id": "T1595", "name": "Active Scanning",            "tactic": "Reconnaissance"},
    "port scan":       {"id": "T1046", "name": "Network Service Discovery",  "tactic": "Discovery"},
    "arp spoofing":    {"id": "T1557", "name": "Adversary-in-the-Middle",    "tactic": "Credential Access"},
    "mqtt":            {"id": "T1071", "name": "Application Layer Protocol", "tactic": "Command & Control"},
    "unknown_anomaly": {"id": "T1190", "name": "Exploit Public-Facing App",  "tactic": "Initial Access"},
}

def lookup_mitre(attack_type: str) -> dict:
    key   = attack_type.lower().replace("_", " ").replace("-", " ")
    match = MITRE_LOCAL.get(key)
    if not match:
        for k, v in MITRE_LOCAL.items():
            if k in key or any(word in key for word in k.split()):
                match = v
                break
    if match:
        return {"status": "ok", "technique": match}
    return {"status": "ok", "technique": {"id": "T0000", "name": "Unknown Technique", "tactic": "Unknown"}}


ATTACK_DETAILS = {
    "ddos-syn_flood":           "SYN flood overwhelms target with half-open TCP connections.",
    "ddos-tcp_flood":           "TCP flood sends massive TCP packets to consume bandwidth.",
    "ddos-icmp_flood":          "ICMP flood saturates target with echo requests.",
    "ddos-udp_flood":           "UDP flood sends large volumes of UDP packets to random ports.",
    "dos-syn_flood":            "Single-source SYN flood targeting IoMT device availability.",
    "dos-tcp_flood":            "Single-source TCP flood against IoMT device.",
    "dos-icmp_flood":           "Single-source ICMP flood against IoMT device.",
    "dos-udp_flood":            "Single-source UDP flood against IoMT device.",
    "recon-ping_sweep":         "Ping sweep discovers live hosts on the network.",
    "recon-vulnerability_scan": "Automated scan identifies open ports and vulnerabilities.",
    "recon-os_scan":            "OS fingerprinting identifies operating system of target.",
    "recon-port_scan":          "Port scan enumerates open services on the target.",
    "spoofing-arp":             "ARP spoofing poisons ARP cache to intercept traffic.",
    "mqtt-malformed_data":      "Malformed MQTT packets attempt to crash or destabilize broker.",
    "mqtt-ddos-connect_flood":  "MQTT connection flood exhausts broker connection pool.",
    "mqtt-ddos-publish_flood":  "MQTT publish flood overwhelms broker message queue.",
    "mqtt-dos-connect_flood":   "Single-source MQTT connection flood from compromised device.",
    "mqtt-dos-publish_flood":   "Single-source MQTT publish flood from compromised device.",
    "unknown_anomaly":          "Unclassified anomaly — possible zero-day.",
}

def get_attack_details(attack_type: str) -> dict:
    key    = attack_type.lower().replace(" ", "_").replace("-", "_")
    detail = None
    for k, v in ATTACK_DETAILS.items():
        if k in key or key in k:
            detail = v
            break
    return {"status": "ok", "attack_type": attack_type,
            "details": detail or "No specific details. Treat as potential zero-day."}


# ── CVSS v3.1 per-attack-type scores (standard-compliant) ────────────────────
CVSS_DB = {
    "ddos":            {"score": 7.5, "severity": "HIGH",     "vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", "rationale": "Network-based, no privileges, high availability impact"},
    "dos":             {"score": 7.5, "severity": "HIGH",     "vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H", "rationale": "Single-source DoS, high availability impact"},
    "mqtt":            {"score": 8.6, "severity": "HIGH",     "vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:L/A:H", "rationale": "Affects MQTT broker confidentiality, integrity, and availability"},
    "spoofing":        {"score": 8.1, "severity": "HIGH",     "vector": "AV:A/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:N", "rationale": "Adjacent network MITM, high confidentiality and integrity impact"},
    "recon":           {"score": 5.3, "severity": "MEDIUM",   "vector": "AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N", "rationale": "Reconnaissance only, low confidentiality impact"},
    "unknown_anomaly": {"score": 7.2, "severity": "HIGH",     "vector": "AV:N/AC:H/PR:N/UI:N/S:U/C:L/I:L/A:H", "rationale": "Unknown vector, high complexity assumed, potential full impact"},
}


def estimate_severity(anomaly_score: float, lgbm_confidence: float,
                      anomaly_threshold: float, attack_type: str = "") -> dict:
    """
    Estimate severity using CVSS v3.1 per-attack-type scores.
    Escalates to CRITICAL if anomaly ratio >= 5x threshold.
    Falls back to rule-based for unrecognised attack types.
    Reference: https://www.first.org/cvss/calculator/3.1
    """
    ratio = anomaly_score / anomaly_threshold if anomaly_threshold > 0 else 1.0

    # Match attack type to CVSS DB
    key  = attack_type.lower()
    cvss = None
    for k, v in CVSS_DB.items():
        if k in key:
            cvss = v
            break

    if cvss:
        score     = cvss["score"]
        severity  = cvss["severity"]
        vector    = cvss["vector"]
        rationale = cvss["rationale"]
    else:
        # Fallback rule-based
        if lgbm_confidence >= 0.95 or ratio >= 3.0:
            severity, score = "CRITICAL", 9.0
        elif lgbm_confidence >= 0.85 or ratio >= 2.0:
            severity, score = "HIGH", 7.5
        elif lgbm_confidence >= 0.70 or ratio >= 1.5:
            severity, score = "MEDIUM", 5.0
        else:
            severity, score = "LOW", 3.0
        vector    = "N/A"
        rationale = "Rule-based fallback — attack type not in CVSS database"

    # Escalate if anomaly ratio is extreme
    if ratio >= 5.0 and severity != "CRITICAL":
        severity   = "CRITICAL"
        score      = min(score + 1.5, 10.0)
        rationale += f" [escalated: anomaly ratio {ratio:.1f}x]"

    return {
        "status":        "ok",
        "severity":      severity,
        "cvss_score":    round(score, 1),
        "cvss_vector":   vector,
        "cvss_version":  "3.1",
        "rationale":     rationale,
        "anomaly_ratio": round(ratio, 2),
    }



MITIGATIONS = {
    "ddos":            ["Enable rate limiting on all network interfaces.",
                        "Deploy upstream traffic scrubbing.",
                        "Block source IPs at firewall level.",
                        "Enable SYN cookies on affected servers."],
    "dos":             ["Block attacking IP at the firewall immediately.",
                        "Enable connection rate limiting.",
                        "Isolate affected IoMT device from the network.",
                        "Review and patch any exposed services."],
    "recon":           ["Block scanning source IP at perimeter firewall.",
                        "Disable unnecessary open ports and services.",
                        "Enable port-scan detection on IDS/IPS.",
                        "Review network segmentation."],
    "spoofing":        ["Enable Dynamic ARP Inspection (DAI) on managed switches.",
                        "Use static ARP entries for critical devices.",
                        "Deploy 802.1X port authentication.",
                        "Monitor ARP tables for anomalies."],
    "mqtt":            ["Enable MQTT authentication and TLS encryption.",
                        "Validate all incoming MQTT payloads at broker.",
                        "Restrict MQTT broker access by IP whitelist.",
                        "Update MQTT broker to latest patched version."],
    "unknown_anomaly": ["Isolate the affected device immediately.",
                        "Capture full packet trace for forensic analysis.",
                        "Check device firmware and apply available patches.",
                        "Monitor for repeated anomalies — possible zero-day.",
                        "Report to security team for deeper investigation."],
}

def suggest_mitigation(attack_type: str) -> dict:
    key   = attack_type.lower()
    steps = None
    for k, v in MITIGATIONS.items():
        if k in key:
            steps = v
            break
    if not steps:
        steps = MITIGATIONS["unknown_anomaly"]
    return {"status": "ok", "attack_type": attack_type, "mitigation_steps": steps}


TOOL_REGISTRY = {
    "search_cve":         search_cve,
    "lookup_mitre":       lookup_mitre,
    "get_attack_details": get_attack_details,
    "estimate_severity":  estimate_severity,
    "suggest_mitigation": suggest_mitigation,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "search_cve",
        "description": "Search local CVE database for matching vulnerabilities.",
        "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}}},
    {"type": "function", "function": {"name": "lookup_mitre",
        "description": "Look up MITRE ATT&CK technique for a given attack type.",
        "parameters": {"type": "object", "properties": {"attack_type": {"type": "string"}}, "required": ["attack_type"]}}},
    {"type": "function", "function": {"name": "get_attack_details",
        "description": "Get description of a known attack type.",
        "parameters": {"type": "object", "properties": {"attack_type": {"type": "string"}}, "required": ["attack_type"]}}},
    {"type": "function", "function": {"name": "estimate_severity",
        "description": "Estimate attack severity from IDS scores.",
        "parameters": {"type": "object", "properties": {
            "anomaly_score": {"type": "number"}, "lgbm_confidence": {"type": "number"},
            "anomaly_threshold": {"type": "number"}},
            "required": ["anomaly_score", "lgbm_confidence", "anomaly_threshold"]}}},
    {"type": "function", "function": {"name": "suggest_mitigation",
        "description": "Get mitigation steps for the detected attack type.",
        "parameters": {"type": "object", "properties": {"attack_type": {"type": "string"}}, "required": ["attack_type"]}}},
]
