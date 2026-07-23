# =============================================================================
# experiments/selective_llm_eval.py — Table 10: Selective LLM invocation
# Compares baseline (invoke-per-alert) vs selective invocation policy.
# Reports: alerts generated, LLM invocations, reduction %, compute time,
#          distinct campaigns, campaigns covered %.
# Usage: python experiments/selective_llm_eval.py
# =============================================================================

import os, sys, time, json, pickle, numpy as np, datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, REPORT_DIR, LLM_HOST, LLM_TIMEOUT

try:
    import requests
    import pandas as pd
    import glob
except ImportError as e:
    print(f"Missing: {e}"); sys.exit(1)

N_RUNS        = 3
WINDOW_S      = 60
LLM_CVSS_THR  = 7.0
LLM_MODEL     = "qwen3:4b"

# ── Load IDS models ───────────────────────────────────────────────────────────

def load_ids():
    from ids.lightgbm_clf import load_model as lgbm_load, predict as lgbm_predict
    from ids.autoencoder  import load_model as ae_load,   predict as ae_predict
    from ids.alert        import build_alert
    from agent.tools      import estimate_severity

    with open(os.path.join(MODEL_DIR,"label_encoder.pkl"),"rb") as f:
        le=pickle.load(f)
    with open(os.path.join(MODEL_DIR,"feature_names.pkl"),"rb") as f:
        feat=pickle.load(f)

    lgbm=lgbm_load(); ae,thr=ae_load()
    return lgbm,ae,thr,le,feat,lgbm_predict,ae_predict,build_alert,estimate_severity


def load_test_data():
    from data.preprocess import preprocess
    _,X_test,_,y_test,le,_=preprocess()
    return X_test, y_test, le


# ── Selective invocation policy ───────────────────────────────────────────────

def should_invoke(alert:dict, estimate_sev_fn) -> bool:
    if alert.get("is_anomaly") and not alert.get("is_known_attack"):
        return True
    if alert.get("attack_type") == "unknown_anomaly":
        return True
    if alert.get("severity") in ["CRITICAL","HIGH"]:
        return True
    try:
        r=estimate_sev_fn(alert.get("anomaly_score",0),
                          alert.get("lgbm_confidence",0),
                          alert.get("anomaly_threshold",1),
                          alert.get("attack_type",""))
        return float(r.get("cvss_score",0)) >= LLM_CVSS_THR
    except:
        return True


# ── Campaign tracker ──────────────────────────────────────────────────────────

def campaign_key(alert:dict) -> str:
    fam=alert.get("attack_type","?").split("-")[0].upper()
    return f"{fam}|{alert.get('src_ip','?')}|{alert.get('dst_ip','?')}"


# ── Simulate LLM call time ────────────────────────────────────────────────────

def simulate_llm_call(alert:dict) -> float:
    """Make a real minimal LLM call and return elapsed time."""
    prompt=(f"/no_think\nAlert: {alert.get('attack_type','unknown')} "
            f"severity={alert.get('severity','?')}. "
            f"One line summary.")
    start=time.time()
    try:
        requests.post(f"{LLM_HOST}/api/chat",timeout=LLM_TIMEOUT,
            json={"model":LLM_MODEL,
                  "messages":[{"role":"user","content":prompt}],
                  "stream":False,
                  "options":{"num_ctx":128,"num_predict":32,"temperature":0.1}})
    except:
        pass
    return round(time.time()-start,2)


# ── One run ───────────────────────────────────────────────────────────────────

def run_one(X_test, lgbm, ae, thr, le, feat,
            lgbm_predict, ae_predict, build_alert, estimate_sev,
            n_samples=200, seed=42):
    np.random.seed(seed)
    indices=np.random.choice(len(X_test),min(n_samples,len(X_test)),replace=False)

    alerts=[]
    for idx in indices:
        features=X_test[idx]
        lr=lgbm_predict(lgbm,features,le)
        ar=ae_predict(ae,thr,features)
        alert=build_alert(features,feat,lr,ar)
        if alert: alerts.append(alert)

    total_alerts=len(alerts)
    if total_alerts==0:
        return None

    # ── Baseline: invoke LLM for every alert ─────────────────────────────────
    baseline_invocations = total_alerts
    baseline_time        = 0.0
    baseline_campaigns   = set()
    for a in alerts:
        baseline_campaigns.add(campaign_key(a))
        baseline_time += simulate_llm_call(a)

    # ── Selective policy ──────────────────────────────────────────────────────
    selective_invocations = 0
    selective_time        = 0.0
    campaigns_seen        = {}   # key → first_time
    campaigns_with_llm    = set()
    all_campaigns         = set()
    base_ts               = time.time()

    for i, a in enumerate(alerts):
        # Simulate timestamps (spread over 120s to create campaigns)
        a_time = base_ts + (i / total_alerts) * 120
        ckey   = campaign_key(a)
        all_campaigns.add(ckey)

        # Campaign deduplication
        last_ts = campaigns_seen.get(ckey, 0)
        is_new  = (a_time - last_ts) > WINDOW_S
        if is_new:
            campaigns_seen[ckey] = a_time
        else:
            continue  # duplicate — log only, no invocation

        # Selective invocation
        if should_invoke(a, estimate_sev):
            selective_invocations += 1
            selective_time        += simulate_llm_call(a)
            campaigns_with_llm.add(ckey)

    distinct_campaigns   = len(all_campaigns)
    campaigns_covered    = round(len(campaigns_with_llm)/
                                  max(distinct_campaigns,1)*100,1)
    invocation_reduction = round((1 - selective_invocations/
                                   max(baseline_invocations,1))*100,1)

    return {
        "total_alerts":          total_alerts,
        "baseline_invocations":  baseline_invocations,
        "selective_invocations": selective_invocations,
        "invocation_reduction":  invocation_reduction,
        "baseline_time_h":       round(baseline_time/3600,4),
        "selective_time_h":      round(selective_time/3600,4),
        "distinct_campaigns":    distinct_campaigns,
        "campaigns_covered_pct": campaigns_covered,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n"+"="*65)
    print("SELECTIVE LLM INVOCATION EVALUATION — Table 10")
    print(f"Runs={N_RUNS} | Window={WINDOW_S}s | CVSS threshold={LLM_CVSS_THR}")
    print("="*65)

    print("\n[Eval] Loading models...")
    lgbm,ae,thr,le,feat,lgbm_predict,ae_predict,build_alert,estimate_sev=load_ids()
    X_test,_,_=load_test_data()
    print(f"[Eval] Test samples: {len(X_test):,}")

    all_runs=[]
    for run in range(N_RUNS):
        seed=42+run*100
        print(f"\n[Eval] Run {run+1}/{N_RUNS} (seed={seed})...")
        r=run_one(X_test,lgbm,ae,thr,le,feat,
                  lgbm_predict,ae_predict,build_alert,estimate_sev,
                  n_samples=200,seed=seed)
        if r:
            all_runs.append(r)
            print(f"  Alerts       : {r['total_alerts']}")
            print(f"  Baseline inv : {r['baseline_invocations']}")
            print(f"  Selective inv: {r['selective_invocations']}")
            print(f"  Reduction    : {r['invocation_reduction']}%")
            print(f"  Campaigns    : {r['distinct_campaigns']}")
            print(f"  Coverage     : {r['campaigns_covered_pct']}%")

    if not all_runs:
        print("[Eval] No results"); return

    def avg(k): return round(np.mean([r[k] for r in all_runs]),2)

    # ── Print Table 10 ────────────────────────────────────────────────────────
    print("\n\n"+"="*80)
    print("Table 10. Selective LLM invocation efficiency (mean over 3 runs)")
    print("="*80)
    print(f"{'Configuration':<28}{'Alerts':>8}{'LLM inv':>9}"
          f"{'Reduction%':>12}{'Time(h)':>9}{'Campaigns':>11}{'Coverage%':>11}")
    print("-"*80)
    print(f"{'Invoke-per-alert (baseline)':<28}"
          f"{avg('total_alerts'):>8.0f}"
          f"{avg('baseline_invocations'):>9.0f}"
          f"{'—':>12}"
          f"{avg('baseline_time_h'):>9.4f}"
          f"{avg('distinct_campaigns'):>11.0f}"
          f"{'100 (by construction)':>11}")
    print(f"{'Selective invocation policy':<28}"
          f"{avg('total_alerts'):>8.0f}"
          f"{avg('selective_invocations'):>9.0f}"
          f"{avg('invocation_reduction'):>11.1f}%"
          f"{avg('selective_time_h'):>9.4f}"
          f"{avg('distinct_campaigns'):>11.0f}"
          f"{avg('campaigns_covered_pct'):>10.1f}%")
    print("="*80)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(REPORT_DIR,exist_ok=True)
    ts=datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    jp=os.path.join(REPORT_DIR,f"selective_llm_{ts}.json")
    mp=os.path.join(REPORT_DIR,f"selective_llm_{ts}.md")

    with open(jp,"w") as f:
        json.dump({"timestamp":datetime.datetime.utcnow().isoformat(),
                   "n_runs":N_RUNS,"window_s":WINDOW_S,
                   "cvss_threshold": LLM_CVSS_THR,
                   "runs":all_runs,
                   "summary":{k:avg(k) for k in all_runs[0].keys()
                               if isinstance(all_runs[0][k],(int,float))}},
                  f,indent=2)

    md ="# Table 10: Selective LLM Invocation Efficiency\n\n"
    md+=f"**Generated:** {ts} | **Runs:** {N_RUNS} | "
    md+=f"**Window:** {WINDOW_S}s | **CVSS threshold:** {LLM_CVSS_THOLD if False else 7.0}\n\n"
    md+=("| Configuration | Alerts generated | LLM invocations | "
         "Invocation reduction (%) | Total LLM compute time | "
         "Distinct campaigns | Campaigns covered (%) |\n")
    md+="|---|---|---|---|---|---|---|\n"
    md+=(f"| Invoke-per-alert (baseline) | {avg('total_alerts'):.0f} | "
         f"{avg('baseline_invocations'):.0f} | — | "
         f"{avg('baseline_time_h'):.4f} h | "
         f"{avg('distinct_campaigns'):.0f} | 100 (by construction) |\n")
    md+=(f"| Selective invocation policy | {avg('total_alerts'):.0f} | "
         f"{avg('selective_invocations'):.0f} | "
         f"{avg('invocation_reduction'):.1f}% | "
         f"{avg('selective_time_h'):.4f} h | "
         f"{avg('distinct_campaigns'):.0f} | "
         f"{avg('campaigns_covered_pct'):.1f}% |\n")

    with open(mp,"w") as f: f.write(md)
    print(f"\n[Eval] Saved:\n  JSON → {jp}\n  MD   → {mp}")


if __name__=="__main__":
    main()
