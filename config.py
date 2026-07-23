# =============================================================================
# config.py — Central configuration for IoT IDS project
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data", "raw")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
REPORT_DIR      = os.path.join(BASE_DIR, "reports")

# ── Dataset ───────────────────────────────────────────────────────────────────
LABEL_COLUMN    = "label"
BENIGN_LABEL    = "Benign"

# ── IDS — LightGBM ────────────────────────────────────────────────────────────
LGBM_MODEL_PATH         = os.path.join(MODEL_DIR, "lgbm_model.pkl")
LGBM_CONFIDENCE_THRESH  = 0.80

# ── IDS — Transformer Autoencoder ─────────────────────────────────────────────
AE_MODEL_PATH           = os.path.join(MODEL_DIR, "autoencoder.pt")
AE_THRESHOLD_PATH       = os.path.join(MODEL_DIR, "ae_threshold.pkl")
AE_EPOCHS               = 30
AE_BATCH_SIZE           = 256
AE_LEARNING_RATE        = 1e-3
AE_ANOMALY_PERCENTILE   = 95

# ── Alert ─────────────────────────────────────────────────────────────────────
ALERT_LOG_PATH  = os.path.join(LOG_DIR, "alerts.jsonl")

# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_MODEL       = "qwen3:4b"
LLM_HOST        = "http://localhost:11434"
LLM_TIMEOUT     = 1600

# ── LLM Analysis stages ───────────────────────────────────────────────────────
SUPERFICIAL_ANALYSIS_ENABLED    = True
DEEP_ANALYSIS_ENABLED           = False   # Phase 3

# ── RAG (Phase 2) ─────────────────────────────────────────────────────────────
RAG_ENABLED     = False
RAG_DB_PATH     = os.path.join(BASE_DIR, "rag", "vectorstore")

# ── Live intelligence (Phase 4) ───────────────────────────────────────────────
LIVE_INTEL_ENABLED      = False
LIVE_INTEL_SCHEDULE_HRS = 24
LIVE_INTEL_LOG_PATH     = os.path.join(LOG_DIR, "live_intel.jsonl")

# ── Tools ─────────────────────────────────────────────────────────────────────
NVD_API_URL     = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_API_KEY     = ""
GITHUB_TOKEN    = ""

# ── CVE local database (Issue 8 — offline CVE retrieval) ──────────────────────
CVE_DB_PATH     = os.path.join(BASE_DIR, "rag", "cve_db", "cve_database.json")

# ── Ensure directories exist ──────────────────────────────────────────────────
for d in [DATA_DIR, MODEL_DIR, LOG_DIR, REPORT_DIR,
          os.path.join(BASE_DIR, "rag", "cve_db")]:
    os.makedirs(d, exist_ok=True)