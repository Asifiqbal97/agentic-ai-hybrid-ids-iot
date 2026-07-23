# =============================================================================
# rag/live_intel.py — Phase 4: Live threat intelligence fetcher
# Sources: NVD API, MITRE ATT&CK STIX, CISA ICS, GitHub Security, Exploit-DB
# Run manually: python rag/live_intel.py
# Or scheduled via cron: 0 0 * * * python /path/to/iot_ids/rag/live_intel.py
# =============================================================================

import os
import re
import sys
import time
import json
import requests
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NVD_API_URL, NVD_API_KEY, GITHUB_TOKEN

# ── Constants ─────────────────────────────────────────────────────────────────
HEADERS_JSON   = {"Accept": "application/json", "User-Agent": "iot-ids-agent/1.0"}
TIMEOUT        = 15
DAYS_LOOKBACK  = 30   # fetch data from last 30 days


# =============================================================================
# SOURCE 1 — NVD API (latest IoT CVEs)
# =============================================================================

def fetch_nvd_latest() -> list:
    """
    Fetch latest IoT/IoMT CVEs from NVD published in last DAYS_LOOKBACK days.
    Returns list of text chunks.
    """
    chunks   = []
    keywords = ["MQTT", "IoT", "IoMT", "medical device", "ICS SCADA",
                "industrial control", "embedded device", "firmware"]
    headers  = {**HEADERS_JSON}
    if NVD_API_KEY:
        headers["apiKey"] = NVD_API_KEY

    # Date range for recent CVEs
    end_date   = datetime.utcnow()
    start_date = end_date - timedelta(days=DAYS_LOOKBACK)
    pub_start  = start_date.strftime("%Y-%m-%dT%H:%M:%S.000")
    pub_end    = end_date.strftime("%Y-%m-%dT%H:%M:%S.000")

    print(f"[LiveIntel] Fetching NVD CVEs (last {DAYS_LOOKBACK} days)...")

    for keyword in keywords:
        try:
            params = {
                "keywordSearch":    keyword,
                "pubStartDate":     pub_start,
                "pubEndDate":       pub_end,
                "resultsPerPage":   10,
            }
            resp = requests.get(NVD_API_URL, params=params,
                                headers=headers, timeout=TIMEOUT)
            if resp.status_code != 200:
                continue

            for item in resp.json().get("vulnerabilities", []):
                cve     = item.get("cve", {})
                cve_id  = cve.get("id", "")
                desc    = cve.get("descriptions", [{}])[0].get("value", "")
                published = cve.get("published", "")[:10]

                metrics = cve.get("metrics", {})
                cvss    = "N/A"
                for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                    if key in metrics:
                        cvss = metrics[key][0]["cvssData"].get("baseScore", "N/A")
                        break

                if desc and len(desc) > 50:
                    chunks.append(
                        f"[LIVE CVE {published}] {cve_id} (CVSS:{cvss}): "
                        f"{desc[:300]}"
                    )
            time.sleep(1)   # NVD rate limit

        except Exception as e:
            print(f"[LiveIntel] NVD error for '{keyword}': {e}")

    print(f"[LiveIntel] NVD: {len(chunks)} chunks fetched")
    return chunks


# =============================================================================
# SOURCE 2 — MITRE ATT&CK STIX API (latest IoT/ICS techniques)
# =============================================================================

MITRE_STIX_URL = "https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json"

def fetch_mitre_live() -> list:
    """
    Fetch latest ICS/IoT ATT&CK techniques from MITRE CTI GitHub repository.
    Returns list of text chunks.
    """
    chunks = []
    print("[LiveIntel] Fetching MITRE ATT&CK ICS techniques...")

    try:
        resp = requests.get(MITRE_STIX_URL, timeout=30)
        if resp.status_code != 200:
            print(f"[LiveIntel] MITRE fetch failed: {resp.status_code}")
            return chunks

        data       = resp.json()
        objects    = data.get("objects", [])
        techniques = [o for o in objects if o.get("type") == "attack-pattern"
                      and not o.get("revoked", False)]

        # IoT/ICS relevant keywords
        iot_keywords = ["network", "denial", "flood", "spoof", "scan",
                        "mqtt", "modbus", "dnp3", "firmware", "exploit",
                        "command", "control", "reconnaissance"]

        for tech in techniques:
            name = tech.get("name", "")
            desc = tech.get("description", "")[:300]
            ext  = tech.get("external_references", [{}])
            tid  = next((r.get("external_id", "") for r in ext
                         if r.get("source_name") == "mitre-attack"), "")

            if any(k in name.lower() or k in desc.lower()
                   for k in iot_keywords):
                chunks.append(
                    f"[LIVE MITRE {tid}] {name}: {desc}"
                )

        print(f"[LiveIntel] MITRE: {len(chunks)} chunks fetched")

    except Exception as e:
        print(f"[LiveIntel] MITRE error: {e}")

    return chunks


# =============================================================================
# SOURCE 3 — CISA ICS Advisories
# =============================================================================

CISA_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"

def fetch_cisa_advisories() -> list:
    """
    Fetch CISA Known Exploited Vulnerabilities relevant to IoT/ICS.
    Returns list of text chunks.
    """
    chunks      = []
    iot_vendors = ["siemens", "schneider", "rockwell", "honeywell",
                   "philips", "ge healthcare", "mqtt", "modbus",
                   "iot", "scada", "industrial"]

    print("[LiveIntel] Fetching CISA known exploited vulnerabilities...")

    try:
        resp = requests.get(CISA_URL, headers=HEADERS_JSON, timeout=TIMEOUT)
        if resp.status_code != 200:
            print(f"[LiveIntel] CISA fetch failed: {resp.status_code}")
            return chunks

        vulns = resp.json().get("vulnerabilities", [])

        for v in vulns:
            vendor  = v.get("vendorProject", "").lower()
            product = v.get("product", "").lower()
            name    = v.get("vulnerabilityName", "")
            desc    = v.get("shortDescription", "")
            cve_id  = v.get("cveID", "")
            due     = v.get("dueDate", "")

            if any(k in vendor or k in product
                   for k in iot_vendors):
                chunks.append(
                    f"[LIVE CISA] {cve_id} — {name} "
                    f"({v.get('vendorProject','')} {v.get('product','')}): "
                    f"{desc[:250]} [Remediation due: {due}]"
                )

        print(f"[LiveIntel] CISA: {len(chunks)} chunks fetched")

    except Exception as e:
        print(f"[LiveIntel] CISA error: {e}")

    return chunks


# =============================================================================
# SOURCE 4 — GitHub Security Advisories (IoT related)
# =============================================================================

GITHUB_ADVISORY_URL = "https://api.github.com/graphql"

GITHUB_QUERY = """
{
  securityAdvisories(first: 20, orderBy: {field: PUBLISHED_AT, direction: DESC},
    ecosystem: null) {
    nodes {
      ghsaId
      summary
      description
      severity
      publishedAt
      vulnerabilities(first: 3) {
        nodes {
          package { name ecosystem }
        }
      }
    }
  }
}
"""

def fetch_github_advisories() -> list:
    """
    Fetch latest security advisories from GitHub Advisory Database.
    Filters for IoT-relevant packages.
    Returns list of text chunks.
    """
    chunks = []

    if not GITHUB_TOKEN:
        print("[LiveIntel] GitHub: No token configured — skipping")
        return chunks

    iot_keywords = ["mqtt", "coap", "zigbee", "zwave", "ble", "lorawan",
                    "modbus", "dnp3", "opcua", "firmware", "embedded",
                    "arduino", "esp32", "raspberry"]

    print("[LiveIntel] Fetching GitHub security advisories...")

    try:
        headers = {
            "Authorization": f"bearer {GITHUB_TOKEN}",
            "Content-Type":  "application/json"
        }
        resp = requests.post(
            GITHUB_ADVISORY_URL,
            json={"query": GITHUB_QUERY},
            headers=headers,
            timeout=TIMEOUT
        )
        if resp.status_code != 200:
            print(f"[LiveIntel] GitHub fetch failed: {resp.status_code}")
            return chunks

        advisories = (resp.json()
                      .get("data", {})
                      .get("securityAdvisories", {})
                      .get("nodes", []))

        for adv in advisories:
            summary  = adv.get("summary", "")
            desc     = adv.get("description", "")[:250]
            severity = adv.get("severity", "")
            ghsa_id  = adv.get("ghsaId", "")
            pub_date = adv.get("publishedAt", "")[:10]

            if any(k in summary.lower() or k in desc.lower()
                   for k in iot_keywords):
                chunks.append(
                    f"[LIVE GitHub {pub_date}] {ghsa_id} ({severity}): "
                    f"{summary}. {desc}"
                )

        print(f"[LiveIntel] GitHub: {len(chunks)} chunks fetched")

    except Exception as e:
        print(f"[LiveIntel] GitHub error: {e}")

    return chunks


# =============================================================================
# SOURCE 5 — Exploit-DB (recent IoT exploits)
# =============================================================================

EXPLOITDB_URL = "https://www.exploit-db.com/search"

def fetch_exploitdb() -> list:
    """
    Fetch recent IoT-related exploits from Exploit-DB.
    Returns list of text chunks.
    """
    chunks      = []
    iot_terms   = ["IoT", "MQTT", "router", "firmware", "camera",
                   "medical device", "SCADA", "ICS"]

    print("[LiveIntel] Fetching Exploit-DB IoT exploits...")

    try:
        for term in iot_terms[:3]:   # limit requests
            params  = {"q": term, "type": "webapps", "verified": "true"}
            headers = {**HEADERS_JSON, "X-Requested-With": "XMLHttpRequest"}
            resp    = requests.get(EXPLOITDB_URL, params=params,
                                   headers=headers, timeout=TIMEOUT)
            if resp.status_code != 200:
                continue

            # Parse JSON response
            data    = resp.json()
            results = data.get("data", [])

            for r in results[:5]:
                title   = r.get("description", "")
                date    = r.get("date_published", "")[:10]
                eid     = r.get("id", "")
                platform= r.get("platform", {}).get("val", "")

                if title:
                    chunks.append(
                        f"[LIVE ExploitDB {date}] EDB-{eid} ({platform}): "
                        f"{title[:250]}"
                    )
            time.sleep(1)

        print(f"[LiveIntel] ExploitDB: {len(chunks)} chunks fetched")

    except Exception as e:
        print(f"[LiveIntel] ExploitDB error: {e}")

    return chunks


# =============================================================================
# MAIN — Fetch all live sources and return combined chunks
# =============================================================================

def fetch_all_live_intel() -> list:
    """
    Fetch from all live sources.
    Returns combined list of text chunks for RAG vector store.
    """
    print(f"\n[LiveIntel] Starting live intelligence fetch — "
          f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    all_chunks = []
    all_chunks.extend(fetch_nvd_latest())
    all_chunks.extend(fetch_mitre_live())
    all_chunks.extend(fetch_cisa_advisories())
    all_chunks.extend(fetch_github_advisories())
    all_chunks.extend(fetch_exploitdb())

    print(f"\n[LiveIntel] Total live chunks fetched: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    chunks = fetch_all_live_intel()
    print(f"\n[LiveIntel] Done. {len(chunks)} chunks ready for RAG.")
