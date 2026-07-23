# =============================================================================
# main.py — End-to-end pipeline entry point
# Usage:
#   python main.py --train          # train IDS models
#   python main.py --run            # run pipeline on test data
#   python main.py --train --run    # train then run
# =============================================================================

import time
import numpy as np
import psutil

import argparse
import pickle
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import MODEL_DIR, LGBM_MODEL_PATH, AE_MODEL_PATH
from data.preprocess  import preprocess, get_normal_data
from ids.lightgbm_clf import train as lgbm_train, load_model as lgbm_load, predict as lgbm_predict
from ids.autoencoder  import train as ae_train,   load_model as ae_load,   predict as ae_predict
from ids.alert        import build_alert
from agent.orchestrator import handle_alert
from reports.writer   import write_report


def train_models():
    """Train LightGBM and Autoencoder on CIC IoMT 2024 dataset."""
    print("\n" + "="*60)
    print("PHASE 1 — Training IDS Models")
    print("="*60)

    X_train, X_test, y_train, y_test, le, scaler = preprocess()

    # Train LightGBM
    lgbm_model = lgbm_train(X_train, y_train, X_test, y_test, le)

    # Train Autoencoder on normal traffic only
    X_normal   = get_normal_data(X_train, y_train, le)
    ae_model, threshold = ae_train(X_normal)

    print("\n[Main] Training complete.")

# ------------------------- newly added

process = psutil.Process()
ram = process.memory_info().rss / (1024**3)
print(f"Peak RAM: {ram:.2f} GB")


def run_pipeline(n_samples: int = 50):
    """Load models and run full pipeline on test data."""
    print("\n" + "="*60)
    print("PHASE 1 — Running IDS + LLM Pipeline")
    print("="*60)

    # ── Load models and preprocessors ─────────────────────────────────────────
    print("[Main] Loading models...")
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)

    lgbm_model              = lgbm_load()
    ae_model, ae_threshold  = ae_load()




    # ── Load test data ─────────────────────────────────────────────────────────
    from data.preprocess import preprocess as _preprocess
    _, X_test, _, y_test, _, _ = _preprocess()

    # ---------------------------------- newly added


    start = time.time()
    for _ in range(1000):
        lgbm_predict(lgbm_model, X_test[0], le)
    elapsed = (time.time() - start) / 1000 * 1000  # ms per flow
    print(f"LightGBM latency: {elapsed:.3f} ms/flow")

    # -------------------------------------- newly added

    start = time.time()
    for _ in range(1000):
        ae_predict(ae_model, ae_threshold, X_test[0])
    elapsed = (time.time() - start) / 1000 * 1000
    print(f"AE latency: {elapsed:.3f} ms/flow")

    # Sample n_samples flows for demonstration
    indices = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)

    alerts_triggered = 0
    reports_saved    = 0

    for i, idx in enumerate(indices):
        features = X_test[idx]

        # ── Run IDS ───────────────────────────────────────────────────────────
        lgbm_result = lgbm_predict(lgbm_model, features, le)
        ae_result   = ae_predict(ae_model, ae_threshold, features)

        # ── Build alert ───────────────────────────────────────────────────────
        alert = build_alert(features, feature_names, lgbm_result, ae_result)

        if alert is None:
            continue  # benign, no alert

        alerts_triggered += 1
        print(f"\n[Main] Flow {i+1}: Alert! — {alert['attack_type']} | {alert['severity']}")

        # ── ML Agent → LLM ────────────────────────────────────────────────────
        agent_result = handle_alert(alert)

        # ── Report ────────────────────────────────────────────────────────────
        report_path  = write_report(agent_result)
        reports_saved += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"Pipeline complete.")
    print(f"  Flows processed : {len(indices)}")
    print(f"  Alerts triggered: {alerts_triggered}")
    print(f"  Reports saved   : {reports_saved}")
    print(f"  Reports folder  : reports/")
    print("="*60)

# ---------------------------- newly added
process = psutil.Process()
ram = process.memory_info().rss / (1024**3)
print(f"Peak RAM: {ram:.2f} GB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT IDS — Agentic LLM Pipeline")
    parser.add_argument("--train", action="store_true", help="Train IDS models")
    parser.add_argument("--run",   action="store_true", help="Run pipeline on test data")
    parser.add_argument("--samples", type=int, default=50, help="Number of test flows to process")
    args = parser.parse_args()

    if not args.train and not args.run:
        parser.print_help()
    if args.train:
        train_models()
    if args.run:
        run_pipeline(n_samples=args.samples)




# --------------------------------------- newly added


with open("rag/vectorstore/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"RAG chunks: {len(chunks)}")


size = os.path.getsize("rag/vectorstore/index.faiss") / (1024**2)
print(f"FAISS index size: {size:.1f} MB")