# =============================================================================
# rag/builder.py — Build FAISS vector store
# Issue 5: full ATT&CK ICS corpus + bulk NVD IoT CVEs
# Phase 4: live intel via build_vectorstore_with_live_intel()
# =============================================================================

import os, json, time, glob, pickle, requests, numpy as np, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORT_DIR, NVD_API_URL, NVD_API_KEY, RAG_DB_PATH

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Install: pip install faiss-cpu sentence-transformers"); sys.exit(1)

EMBED_MODEL   = "all-MiniLM-L6-v2"
MITRE_ICS_URL = "https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json"

NVD_IOT_KEYWORDS = [
    "MQTT", "IoT", "IoMT", "medical device", "ICS", "SCADA",
    "industrial control", "embedded device", "firmware vulnerability",
    "router vulnerability", "smart home", "wearable device",
    "Modbus", "DNP3", "Zigbee", "BLE vulnerability", "LoRaWAN",
    "CoAP", "OPC-UA", "network camera", "IP camera",
]

OWASP_CHUNKS = [
    "OWASP IoT 1: Weak Guessable or Hardcoded Passwords — use of easily brute forced credentials including backdoors in firmware.",
    "OWASP IoT 2: Insecure Network Services — unneeded or insecure network services running on device, especially exposed to internet.",
    "OWASP IoT 3: Insecure Ecosystem Interfaces — insecure web, backend API, cloud, or mobile interfaces allowing device compromise.",
    "OWASP IoT 4: Lack of Secure Update Mechanism — lack of firmware validation, secure delivery, and anti-rollback mechanisms.",
    "OWASP IoT 5: Use of Insecure or Outdated Components — deprecated software components that could allow device compromise.",
    "OWASP IoT 6: Insufficient Privacy Protection — user personal information stored insecurely or used without permission.",
    "OWASP IoT 7: Insecure Data Transfer and Storage — lack of encryption of sensitive data at rest or in transit.",
    "OWASP IoT 8: Lack of Device Management — no asset management, update management, or secure decommissioning.",
    "OWASP IoT 9: Insecure Default Settings — devices shipped with insecure defaults lacking ability to harden.",
    "OWASP IoT 10: Lack of Physical Hardening — no physical hardening allowing attackers to gain sensitive information.",
]

IOMT_CHUNKS = [
    "MQTT Security: Enable TLS, certificate authentication, ACLs, rate limiting on CONNECT and PUBLISH. Keep broker updated.",
    "MQTT DDoS Protection: Configure connection limits per client, message size restrictions, publish rate limiting, MQTT firewall.",
    "IoMT Segmentation: Isolate on dedicated VLAN, micro-segmentation for critical devices, zero-trust architecture.",
    "ARP Spoofing Prevention: Enable Dynamic ARP Inspection, static ARP for critical devices, 802.1X authentication.",
    "IoMT Device Hardening: Change default credentials, disable unnecessary services, enable encryption, apply firmware updates.",
    "Zero-Day Response: Isolate anomalous devices, capture packet trace, alert team, submit samples to threat intelligence.",
    "CIC IoMT 2024: WiFi and MQTT IoMT traffic, 18 attack types, DDoS/DoS/Recon/Spoofing/MQTT, 45 flow-based features.",
    "Flow Features: ICMP=ICMP floods, SYN=SYN floods, high Rate=volume attacks, low IAT=rapid packets, Variance=pattern uniformity.",
    "CVSS Severity: CRITICAL 9-10 patient risk, HIGH 7-8.9 disruption, MEDIUM 4-6.9 limited, LOW 0.1-3.9 minimal.",
    "DDoS Response: Identify type, isolate segment, rate limit, notify clinical staff, preserve evidence, restore services.",
]

# ── Issue 5: Full ATT&CK ICS ──────────────────────────────────────────────────

STATIC_MITRE = [
    "T1498 Network Denial of Service [impact]: DDoS targeting IoMT device availability and network communications.",
    "T1499 Endpoint Denial of Service [impact]: Service exhaustion on MQTT brokers and IoT gateways via flood attacks.",
    "T1595 Active Scanning [reconnaissance]: Network scanning to discover IoMT devices, open ports, firmware versions.",
    "T1595.002 Vulnerability Scanning [reconnaissance]: Automated probing of IoMT devices for CVE-specific weaknesses.",
    "T1592 Gather Victim Host Information [reconnaissance]: OS and firmware fingerprinting of medical IoT devices.",
    "T1046 Network Service Discovery [discovery]: Port scanning to enumerate IoMT device services including MQTT and HTTP.",
    "T1557 Adversary-in-the-Middle [credential-access]: ARP spoofing to intercept IoMT device communications and patient data.",
    "T1557.002 ARP Cache Poisoning [credential-access]: Poisoning ARP caches to enable MITM on clinical IoT networks.",
    "T1190 Exploit Public-Facing Application [initial-access]: Exploiting MQTT, HTTP, CoAP interfaces on IoT devices.",
    "T1071 Application Layer Protocol [command-and-control]: Using MQTT as covert C2 channel for compromised IoT devices.",
    "T1078 Valid Accounts [initial-access]: Abusing default credentials on IoMT devices for unauthorized access.",
    "T1565 Data Manipulation [impact]: Tampering with patient vital signs data transmitted over IoMT network.",
]

def fetch_full_attck_ics() -> list:
    """Fetch full MITRE ATT&CK ICS corpus. Falls back to static if unavailable."""
    chunks = []
    print("[RAG] Fetching full MITRE ATT&CK ICS corpus...")
    try:
        resp = requests.get(MITRE_ICS_URL, timeout=30)
        if resp.status_code != 200:
            print("[RAG] ATT&CK ICS unavailable — using static chunks")
            return STATIC_MITRE

        objects    = resp.json().get("objects", [])
        techniques = [o for o in objects
                      if o.get("type") == "attack-pattern"
                      and not o.get("revoked", False)
                      and not o.get("x_mitre_deprecated", False)]

        for tech in techniques:
            name   = tech.get("name", "")
            desc   = tech.get("description", "")
            refs   = tech.get("external_references", [])
            tid    = next((r.get("external_id","") for r in refs
                           if r.get("source_name") == "mitre-attack"), "")
            phases = [p.get("phase_name","")
                      for p in tech.get("kill_chain_phases", [])]
            if name and desc:
                chunks.append(f"{tid} {name} [{','.join(phases)}]: {desc[:350]}")

        print(f"[RAG] ATT&CK ICS: {len(chunks)} techniques")
    except Exception as e:
        print(f"[RAG] ATT&CK ICS error: {e} — using static")
        return STATIC_MITRE

    return chunks if chunks else STATIC_MITRE


# ── Issue 5: Bulk NVD CVEs ────────────────────────────────────────────────────

def fetch_bulk_nvd_cves(max_per_keyword: int = 50) -> list:
    """Fetch large volume of IoT CVEs across many keywords."""
    chunks  = []
    seen    = set()
    headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}
    print(f"[RAG] Fetching bulk NVD CVEs ({len(NVD_IOT_KEYWORDS)} keywords, {max_per_keyword} max each)...")

    for keyword in NVD_IOT_KEYWORDS:
        fetched = 0
        start   = 0
        while fetched < max_per_keyword:
            try:
                resp = requests.get(NVD_API_URL,
                    params={"keywordSearch": keyword, "resultsPerPage": 20, "startIndex": start},
                    headers=headers, timeout=15)
                if resp.status_code != 200:
                    break
                data  = resp.json()
                items = data.get("vulnerabilities", [])
                if not items:
                    break
                for item in items:
                    cve    = item.get("cve", {})
                    cve_id = cve.get("id", "")
                    if cve_id in seen:
                        continue
                    seen.add(cve_id)
                    desc = cve.get("descriptions", [{}])[0].get("value", "")
                    pub  = cve.get("published", "")[:10]
                    metrics = cve.get("metrics", {})
                    cvss = "N/A"
                    for key in ["cvssMetricV31","cvssMetricV30","cvssMetricV2"]:
                        if key in metrics:
                            cvss = metrics[key][0]["cvssData"].get("baseScore","N/A")
                            break
                    if desc and len(desc) > 40:
                        chunks.append(f"[CVE {pub}] {cve_id} (CVSS:{cvss}): {desc[:300]}")
                    fetched += 1
                total = data.get("totalResults", 0)
                start += 20
                if start >= min(total, max_per_keyword):
                    break
                time.sleep(0.6)
            except Exception as e:
                print(f"[RAG] NVD error '{keyword}': {e}")
                break
        print(f"  {keyword}: {fetched} CVEs")

    print(f"[RAG] Bulk NVD: {len(chunks)} unique CVE chunks")
    return chunks


def load_phase1_report_chunks() -> list:
    chunks = []
    for path in glob.glob(os.path.join(REPORT_DIR, "*.json")):
        try:
            with open(path) as f:
                report = json.load(f)
            alert  = report.get("alert", {})
            sa     = report.get("superficial_analysis", {})
            answer = sa.get("final_answer", "") if sa else ""
            if answer and len(answer) > 100:
                chunks.append(
                    f"Attack: {alert.get('attack_type','unknown')} | "
                    f"Severity: {alert.get('severity','N/A')} | "
                    f"Analysis: {answer[:400]}"
                )
        except Exception:
            continue
    print(f"[RAG] Phase 1 reports: {len(chunks)} chunks")
    return chunks


def _embed_and_save(all_chunks: list):
    all_chunks = list(dict.fromkeys(all_chunks))  # deduplicate
    print(f"[RAG] Total unique chunks: {len(all_chunks)}")
    print(f"[RAG] Embedding with {EMBED_MODEL}...")
    embedder   = SentenceTransformer(EMBED_MODEL)
    embeddings = embedder.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    os.makedirs(RAG_DB_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(RAG_DB_PATH, "index.faiss"))
    with open(os.path.join(RAG_DB_PATH, "chunks.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)
    with open(os.path.join(RAG_DB_PATH, "embedder_name.txt"), "w") as f:
        f.write(EMBED_MODEL)
    print(f"[RAG] Vector store saved → {RAG_DB_PATH}")
    print(f"[RAG] Index: {index.ntotal} vectors, dim={dim}")


def build_vectorstore():
    """Phase 2 + Issue 5: scaled static RAG."""
    print("[RAG] Building scaled vector store...")
    all_chunks = []
    all_chunks.extend(fetch_full_attck_ics())
    all_chunks.extend(OWASP_CHUNKS)
    all_chunks.extend(IOMT_CHUNKS)
    all_chunks.extend(fetch_bulk_nvd_cves(max_per_keyword=50))
    all_chunks.extend(load_phase1_report_chunks())
    _embed_and_save(all_chunks)


def build_vectorstore_with_live_intel():
    """Phase 4: static + live intel."""
    from config import LIVE_INTEL_ENABLED, LIVE_INTEL_LOG_PATH
    print("[RAG] Building vector store with live intelligence...")
    all_chunks = []
    all_chunks.extend(fetch_full_attck_ics())
    all_chunks.extend(OWASP_CHUNKS)
    all_chunks.extend(IOMT_CHUNKS)
    all_chunks.extend(fetch_bulk_nvd_cves(max_per_keyword=50))
    all_chunks.extend(load_phase1_report_chunks())
    if LIVE_INTEL_ENABLED:
        from rag.live_intel import fetch_all_live_intel
        live_chunks = fetch_all_live_intel()
        all_chunks.extend(live_chunks)
        import datetime
        os.makedirs(os.path.dirname(LIVE_INTEL_LOG_PATH), exist_ok=True)
        with open(LIVE_INTEL_LOG_PATH, "a") as f:
            f.write(json.dumps({
                "timestamp":    datetime.datetime.utcnow().isoformat(),
                "live_chunks":  len(live_chunks),
                "total_chunks": len(all_chunks)
            }) + "\n")
    _embed_and_save(all_chunks)


if __name__ == "__main__":
    from config import LIVE_INTEL_ENABLED
    if LIVE_INTEL_ENABLED:
        build_vectorstore_with_live_intel()
    else:
        build_vectorstore()
