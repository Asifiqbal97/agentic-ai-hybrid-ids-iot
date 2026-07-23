# =============================================================================
# rag/scheduler.py — Phase 4: Automatic RAG rebuild scheduler
# Rebuilds vector store every LIVE_INTEL_SCHEDULE_HRS hours
# Usage: python rag/scheduler.py
# Or via cron: 0 0 * * * python /path/to/iot_ids/rag/scheduler.py
# =============================================================================

import os
import sys
import time
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LIVE_INTEL_SCHEDULE_HRS, LIVE_INTEL_ENABLED

def run_scheduler():
    """Run continuous RAG rebuild loop."""

    if not LIVE_INTEL_ENABLED:
        print("[Scheduler] LIVE_INTEL_ENABLED = False in config.py")
        print("[Scheduler] Set LIVE_INTEL_ENABLED = True to enable.")
        return

    print(f"[Scheduler] Starting — rebuild every {LIVE_INTEL_SCHEDULE_HRS} hours")

    while True:
        print(f"\n[Scheduler] Rebuilding RAG — "
              f"{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

        try:
            from rag.builder import build_vectorstore_with_live_intel
            build_vectorstore_with_live_intel()
            print("[Scheduler] RAG rebuild complete.")
        except Exception as e:
            print(f"[Scheduler] Rebuild failed: {e}")

        # Wait for next rebuild
        sleep_secs = LIVE_INTEL_SCHEDULE_HRS * 3600
        print(f"[Scheduler] Next rebuild in {LIVE_INTEL_SCHEDULE_HRS} hours...")
        time.sleep(sleep_secs)


if __name__ == "__main__":
    run_scheduler()
