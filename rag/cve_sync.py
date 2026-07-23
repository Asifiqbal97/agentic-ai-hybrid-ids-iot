# =============================================================================
# rag/cve_sync.py — Download IoT CVEs from NVD and store locally
# Run once to build local DB, then periodically to update:
#   python rag/cve_sync.py
# Replaces inference-time NVD API calls — fully offline after sync
# =============================================================================

import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NVD_API_URL, NVD_API_KEY, CVE_DB_PATH

# IoT-relevant keywords for CVE filtering
IOT_KEYWORDS = [
    "MQTT", "IoT", "IoMT", "medical device", "ICS", "SCADA",
    "industrial control", "embedded", "firmware", "router",
    "camera", "sensor", "Modbus", "DNP3", "Zigbee", "BLE",
    "LoRaWAN", "CoAP", "OPC-UA", "smart home", "wearable"
]

RESULTS_PER_PAGE = 20   # NVD API max per request
MAX_CVES_PER_KW  = 50   # max CVEs per keyword


def fetch_cves_for_keyword(keyword: str, headers: dict) -> list:
    """Fetch CVEs for one keyword from NVD API."""
    cves = []
    start_index = 0

    while len(cves) < MAX_CVES_PER_KW:
        try:
            params = {
                "keywordSearch":  keyword,
                "resultsPerPage": RESULTS_PER_PAGE,
                "startIndex":     start_index,
            }
            resp = requests.get(NVD_API_URL, params=params,
                                headers=headers, timeout=15)
            if resp.status_code != 200:
                break

            data  = resp.json()
            items = data.get("vulnerabilities", [])
            if not items:
                break

            for item in items:
                cve     = item.get("cve", {})
                cve_id  = cve.get("id", "")
                desc    = cve.get("descriptions", [{}])[0].get("value", "")
                published = cve.get("published", "")[:10]
                modified  = cve.get("lastModified", "")[:10]

                # Extract CVSS score
                metrics = cve.get("metrics", {})
                cvss    = "N/A"
                severity = "UNKNOWN"
                for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                    if key in metrics:
                        cvss     = metrics[key][0]["cvssData"].get("baseScore", "N/A")
                        severity = metrics[key][0]["cvssData"].get("baseSeverity", "UNKNOWN")
                        break

                # Extract CWEs
                weaknesses = cve.get("weaknesses", [])
                cwes = []
                for w in weaknesses:
                    for d in w.get("description", []):
                        if d.get("value", "").startswith("CWE-"):
                            cwes.append(d["value"])

                if cve_id and desc and len(desc) > 30:
                    cves.append({
                        "cve_id":    cve_id,
                        "cvss":      cvss,
                        "severity":  severity,
                        "published": published,
                        "modified":  modified,
                        "desc":      desc[:500],
                        "cwes":      cwes,
                        "keyword":   keyword,
                    })

            total = data.get("totalResults", 0)
            start_index += RESULTS_PER_PAGE
            if start_index >= min(total, MAX_CVES_PER_KW):
                break

            time.sleep(0.6)   # NVD rate limit

        except Exception as e:
            print(f"[CVESync] Error fetching '{keyword}': {e}")
            break

    return cves


def sync_cves():
    """
    Download IoT CVEs from NVD and save to local JSON database.
    Merges with existing entries — no duplicates.
    """
    print(f"[CVESync] Starting CVE sync — "
          f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # Load existing DB if present
    existing = {}
    if os.path.exists(CVE_DB_PATH):
        with open(CVE_DB_PATH) as f:
            data = json.load(f)
            existing = {e["cve_id"]: e for e in data.get("cves", [])}
        print(f"[CVESync] Existing DB: {len(existing)} CVEs")

    headers = {"apiKey": NVD_API_KEY} if NVD_API_KEY else {}
    new_count = 0

    for keyword in IOT_KEYWORDS:
        print(f"[CVESync] Fetching: {keyword}...")
        cves = fetch_cves_for_keyword(keyword, headers)
        for cve in cves:
            if cve["cve_id"] not in existing:
                existing[cve["cve_id"]] = cve
                new_count += 1
        print(f"  → {len(cves)} fetched, {new_count} new total")

    # Save updated DB
    os.makedirs(os.path.dirname(CVE_DB_PATH), exist_ok=True)
    db = {
        "last_synced": datetime.utcnow().isoformat(),
        "total":       len(existing),
        "cves":        list(existing.values())
    }
    with open(CVE_DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

    print(f"\n[CVESync] Sync complete.")
    print(f"  Total CVEs in DB : {len(existing)}")
    print(f"  New CVEs added   : {new_count}")
    print(f"  DB saved → {CVE_DB_PATH}")
    return len(existing)


if __name__ == "__main__":
    sync_cves()
