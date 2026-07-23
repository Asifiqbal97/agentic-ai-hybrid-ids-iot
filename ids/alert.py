# =============================================================================
# ids/alert.py — Build structured AlertObject from IDS outputs
# AlertObject is the contract between IDS and ML Agent.
# Fields are designed to support Phase 2 (deep analysis) without changes.
# =============================================================================

import json
import datetime
import os
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ALERT_LOG_PATH, LGBM_CONFIDENCE_THRESH


def build_alert(
    features: np.ndarray,
    feature_names: list,
    lgbm_result: dict,
    ae_result: dict,
    meta: dict = None
) -> dict | None:
    """
    Merge LightGBM + Autoencoder results into one AlertObject.
    Returns None if traffic is benign and no anomaly detected.

    meta: optional dict with src_ip, dst_ip, protocol etc. from raw flow.
    """
    is_known_attack = lgbm_result["lgbm_is_attack"]
    is_anomaly      = ae_result["is_anomaly"]

    # Only raise alert if either model flags the flow
    if not is_known_attack and not is_anomaly:
        return None

    # Determine attack type label
    if is_known_attack:
        attack_type = lgbm_result["lgbm_label"]
    else:
        attack_type = "unknown_anomaly"  # zero-day candidate

    # Determine severity
    severity = _estimate_severity(lgbm_result["lgbm_confidence"], ae_result["anomaly_score"],
                                   ae_result["anomaly_threshold"])

    # Top contributing features (highest values in normalized vector)
    top_features = _top_features(features, feature_names, top_n=5)

    alert = {
        # ── Core fields (Phase 1) ──────────────────────────────────────────
        "timestamp":         datetime.datetime.utcnow().isoformat(),
        "attack_type":       attack_type,
        "is_known_attack":   is_known_attack,
        "is_anomaly":        is_anomaly,
        "severity":          severity,

        # ── LightGBM output ───────────────────────────────────────────────
        "lgbm_label":        lgbm_result["lgbm_label"],
        "lgbm_confidence":   lgbm_result["lgbm_confidence"],

        # ── Autoencoder output ────────────────────────────────────────────
        "anomaly_score":     ae_result["anomaly_score"],
        "anomaly_threshold": ae_result["anomaly_threshold"],

        # ── Flow metadata (from raw data if available) ────────────────────
        "src_ip":            (meta or {}).get("src_ip", "unknown"),
        "dst_ip":            (meta or {}).get("dst_ip", "unknown"),
        "protocol":          (meta or {}).get("protocol", "unknown"),
        "top_features":      top_features,

        # ── Phase 2 placeholders (Deep Analysis) — do not remove ──────────
        "log_file_path":     None,   # Phase 2: path to session log
        "session_id":        None,   # Phase 2: session identifier
        "deep_analysis":     None,   # Phase 2: filled by deep analysis stage
    }

    _log_alert(alert)
    return alert


def _estimate_severity(confidence: float, anomaly_score: float, threshold: float) -> str:
    """Simple rule-based severity estimation."""
    ratio = anomaly_score / threshold if threshold > 0 else 1.0
    if confidence >= 0.95 or ratio >= 3.0:
        return "CRITICAL"
    elif confidence >= 0.85 or ratio >= 2.0:
        return "HIGH"
    elif confidence >= 0.70 or ratio >= 1.5:
        return "MEDIUM"
    else:
        return "LOW"


def _top_features(features: np.ndarray, feature_names: list, top_n: int = 5) -> dict:
    """Return top N features by normalized value."""
    indices = np.argsort(features)[-top_n:][::-1]
    return {feature_names[i]: round(float(features[i]), 4) for i in indices}


def _log_alert(alert: dict):
    """Append alert to JSONL log file."""
    os.makedirs(os.path.dirname(ALERT_LOG_PATH), exist_ok=True)
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(json.dumps(alert) + "\n")


if __name__ == "__main__":
    # Quick test with dummy data
    import numpy as np
    features      = np.random.rand(46)
    feature_names = [f"feature_{i}" for i in range(46)]
    lgbm_result   = {"lgbm_label": "DDoS-SYN_Flood", "lgbm_confidence": 0.93, "lgbm_is_attack": True}
    ae_result     = {"anomaly_score": 0.042, "anomaly_threshold": 0.015, "is_anomaly": True}

    alert = build_alert(features, feature_names, lgbm_result, ae_result)
    print(json.dumps(alert, indent=2))
