# =============================================================================
# experiments/known_attack_eval.py — Table 5: Known-attack detection
# Reports per-class: LightGBM (Prec/Rec/F1), AE (mean recon error, detection
# rate @ tau), ROC-AUC (family vs benign), Hybrid detection rate.
# Usage: python experiments/known_attack_eval.py
# =============================================================================

import os, sys, json, numpy as np, datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORT_DIR

try:
    import lightgbm as lgb
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  accuracy_score, roc_auc_score,
                                  classification_report)
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd, glob
except ImportError as e:
    print(f"Missing: {e}"); sys.exit(1)

N_SEEDS  = 3
SEEDS    = [42, 123, 999]
AE_EPOCHS= 20

FAMILY_MAP = {
    "TCP_IP-DDoS-ICMP1":       "DDoS",
    "TCP_IP-DDoS-ICMP2":       "DDoS",
    "MQTT-DDoS-Connect_Flood": "DDoS",
    "MQTT-DDoS-Publish_Flood": "DDoS",
    "MQTT-DoS-Connect_Flood":  "DoS",
    "MQTT-DoS-Publish_Flood":  "DoS",
    "MQTT-Malformed_Data":     "MQTT",
    "Recon-OS_Scan":           "Reconnaissance",
    "Recon-Ping_Sweep":        "Reconnaissance",
    "Recon-Port_Scan":         "Reconnaissance",
    "Recon-VulScan":           "Reconnaissance",
    "ARP_Spoofing":            "Spoofing-ARP",
    "Benign":                  "Benign",
}
FAMILIES = ["DDoS","DoS","Reconnaissance","MQTT","Spoofing-ARP","Benign"]


# ── Autoencoder ───────────────────────────────────────────────────────────────
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                          dim_feedforward=128, batch_first=True)
        self.encoder     = nn.TransformerEncoder(enc, num_layers=num_layers)
        dec = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                          dim_feedforward=128, batch_first=True)
        self.decoder     = nn.TransformerEncoder(dec, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return self.output_proj(x).squeeze(1)


# ── Data loading ──────────────────────────────────────────────────────────────
def _norm_label(label):
    for s in ["_train.pcap","_test.pcap","_train","_test",".pcap"]:
        if label.endswith(s): label=label[:-len(s)]; break
    return "Benign" if "benign" in label.lower() else label


def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TR   = os.path.join(base,"data","raw","train_csvs")
    TS   = os.path.join(base,"data","raw","test_csvs")
    frames=[]
    for folder,split in [(TR,"train"),(TS,"test")]:
        for path in sorted(glob.glob(os.path.join(folder,"*.csv"))):
            label=_norm_label(os.path.splitext(os.path.basename(path))[0])
            df=pd.read_csv(path,low_memory=False)
            df.columns=df.columns.str.strip()
            df["label"]=label
            df["family"]=FAMILY_MAP.get(label,"Other")
            df["split"]=split
            df.replace([np.inf,-np.inf],np.nan,inplace=True)
            df.dropna(inplace=True)
            frames.append(df)
    data=pd.concat(frames,ignore_index=True)
    print(f"[Eval] Loaded {len(data):,} flows")
    return data


# ── Training ──────────────────────────────────────────────────────────────────
def train_lgbm(X,y,seed=42):
    m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.05,
                          num_leaves=63,random_state=seed,n_jobs=-1,verbose=-1)
    m.fit(X,y); return m


def train_ae(X_normal,seed=42,epochs=AE_EPOCHS):
    torch.manual_seed(seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=TransformerAutoencoder(X_normal.shape[1]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    loss_fn=nn.MSELoss()
    loader=DataLoader(TensorDataset(torch.tensor(X_normal,dtype=torch.float32)),
                      batch_size=256,shuffle=True)
    model.train()
    for ep in range(epochs):
        for (b,) in loader:
            b=b.to(device)
            l=loss_fn(model(b),b)
            opt.zero_grad(); l.backward(); opt.step()
        if (ep+1)%5==0: print(f"  AE ep {ep+1}/{epochs}")
    model.eval()
    X_t=torch.tensor(X_normal,dtype=torch.float32).to(device)
    with torch.no_grad():
        errs=torch.mean((model(X_t)-X_t)**2,dim=1).cpu().numpy()
    thr=float(np.percentile(errs,95))
    return model,thr,device


def ae_recon_errors(model,X,device):
    model.eval()
    X_t=torch.tensor(X,dtype=torch.float32).to(device)
    with torch.no_grad():
        return torch.mean((model(X_t)-X_t)**2,dim=1).cpu().numpy()


# ── One seed run ──────────────────────────────────────────────────────────────
def run_one_seed(data, fcols, seed):
    np.random.seed(seed)

    train_df = data[data["split"]=="train"]
    test_df  = data[data["split"]=="test"]

    # ── Scale ─────────────────────────────────────────────────────────────────
    sc      = MinMaxScaler()
    X_train = sc.fit_transform(train_df[fcols])
    X_test  = sc.transform(test_df[fcols])

    # ── Encode labels ──────────────────────────────────────────────────────────
    le = LabelEncoder(); le.fit(data["label"])
    y_train = le.transform(train_df["label"])
    y_test  = le.transform(test_df["label"])

    # ── Benign index ──────────────────────────────────────────────────────────
    bidx = le.transform(["Benign"])[0]

    # ── Train LightGBM ────────────────────────────────────────────────────────
    lgbm    = train_lgbm(X_train, y_train, seed)
    y_pred  = lgbm.predict(X_test)
    y_proba = lgbm.predict_proba(X_test)

    # ── Train AE on benign only ───────────────────────────────────────────────
    # Use 80% benign train for AE training, 20% for threshold calibration
    X_benign = X_train[(train_df["label"]=="Benign").values]
    X_ae_tr, X_ae_val = train_test_split(X_benign, test_size=0.2,
                                          random_state=seed)
    ae, _, device = train_ae(X_ae_tr, seed, AE_EPOCHS)

    # Set threshold on validation benign split (disjoint from train)
    val_errs = ae_recon_errors(ae, X_ae_val, device)
    thr      = float(np.percentile(val_errs, 95))
    print(f"  AE threshold (seed {seed}): {thr:.6f}")

    # Benign mean reconstruction error (reference = 1.00)
    benign_mean_err = float(val_errs.mean())

    # AE errors on test set
    test_errs = ae_recon_errors(ae, X_test, device)
    ae_flags  = test_errs > thr

    # ── Per-family metrics ────────────────────────────────────────────────────
    results = {}
    test_labels  = test_df["label"].values
    test_families= test_df["family"].values

    for fam in FAMILIES:
        fam_mask = test_families == fam
        if fam_mask.sum() == 0:
            continue

        y_true_fam   = y_test[fam_mask]
        y_pred_fam   = y_pred[fam_mask]
        ae_flags_fam = ae_flags[fam_mask]
        ae_errs_fam  = test_errs[fam_mask]

        # LightGBM metrics (per family = aggregate of constituent classes)
        prec = precision_score(y_true_fam, y_pred_fam, average="weighted",
                               zero_division=0)
        rec  = recall_score(y_true_fam, y_pred_fam, average="weighted",
                            zero_division=0)
        f1   = f1_score(y_true_fam, y_pred_fam, average="weighted",
                        zero_division=0)
        supp = int(fam_mask.sum())

        # AE mean recon error (normalised by benign mean)
        mean_err_ratio = float(ae_errs_fam.mean()) / benign_mean_err \
                         if benign_mean_err > 0 else 0.0

        # AE detection rate @ tau
        if fam == "Benign":
            ae_det = float(ae_flags_fam.sum()) / len(ae_flags_fam) * 100  # FPR
        else:
            ae_det = float(ae_flags_fam.sum()) / len(ae_flags_fam) * 100

        # ROC-AUC: family vs benign
        # Combine family flows + benign flows, binary label
        benign_mask = test_families == "Benign"
        combined_mask = fam_mask | benign_mask
        if fam != "Benign" and combined_mask.sum() > 0:
            y_roc   = (test_families[combined_mask] == fam).astype(int)
            sc_roc  = test_errs[combined_mask]
            roc_auc = roc_auc_score(y_roc, sc_roc) \
                      if len(np.unique(y_roc)) > 1 else 0.0
        else:
            roc_auc = 0.0

        # Hybrid detection rate (LGBM OR AE)
        lgbm_atk = y_pred_fam != bidx
        if fam == "Benign":
            hybrid = lgbm_atk | ae_flags_fam  # FPR for benign
        else:
            hybrid = lgbm_atk | ae_flags_fam
        hybrid_rate = float(hybrid.sum()) / len(hybrid) * 100

        results[fam] = {
            "support":        supp,
            "lgbm_precision": round(prec, 4),
            "lgbm_recall":    round(rec,  4),
            "lgbm_f1":        round(f1,   4),
            "ae_mean_err_x":  round(mean_err_ratio, 3),
            "ae_detection":   round(ae_det, 2),
            "roc_auc":        round(roc_auc, 4),
            "hybrid_rate":    round(hybrid_rate, 2),
            "threshold":      round(thr, 6),
            "benign_mean_err":round(benign_mean_err, 6),
        }

    # Overall LightGBM metrics
    overall_acc    = accuracy_score(y_test, y_pred)
    overall_macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    return results, round(overall_acc*100,2), round(overall_macro_f1*100,2)


# ── Aggregate over seeds ──────────────────────────────────────────────────────
def mean_std(values):
    return f"{np.mean(values):.2f} ± {np.std(values):.2f}"


def run_evaluation():
    print("\n"+"="*65)
    print("KNOWN-ATTACK EVALUATION — Table 5")
    print(f"Seeds={SEEDS}")
    print("="*65)

    data = load_data()
    fcols = [c for c in data.columns
             if c not in ["label","family","split"]
             and data[c].dtype in [np.float64,np.float32,np.int64,np.int32]]
    print(f"[Eval] Features: {len(fcols)}")

    all_results = []
    all_acc, all_f1 = [], []

    for seed in SEEDS:
        print(f"\n[Eval] Seed {seed}...")
        res, acc, mf1 = run_one_seed(data, fcols, seed)
        all_results.append(res)
        all_acc.append(acc)
        all_f1.append(mf1)

    # Aggregate
    final = {}
    for fam in FAMILIES:
        keys = ["lgbm_precision","lgbm_recall","lgbm_f1","ae_mean_err_x",
                "ae_detection","roc_auc","hybrid_rate"]
        final[fam] = {}
        for k in keys:
            vals = [r[fam][k] for r in all_results if fam in r]
            final[fam][k] = mean_std(vals)
        final[fam]["support"] = all_results[0].get(fam,{}).get("support",0)

    # Macro averages (over attack families only, excluding Benign)
    atk_fams = [f for f in FAMILIES if f != "Benign"]
    macro = {}
    for k in ["lgbm_recall","lgbm_f1","hybrid_rate"]:
        vals_per_seed = []
        for res in all_results:
            seed_vals = [res[f][k] for f in atk_fams if f in res]
            vals_per_seed.append(np.mean(seed_vals))
        macro[k] = mean_std(vals_per_seed)

    # AE macro detection
    ae_vals = []
    for res in all_results:
        ae_vals.append(np.mean([res[f]["ae_detection"] for f in atk_fams if f in res]))
    macro["ae_detection"] = mean_std(ae_vals)

    roc_vals = []
    for res in all_results:
        roc_vals.append(np.mean([res[f]["roc_auc"] for f in atk_fams if f in res]))
    macro["roc_auc"] = mean_std(roc_vals)

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n\n"+"="*110)
    print("Table 5. Known-attack detection performance (mean ± std over 3 runs)")
    print("="*110)
    print(f"{'Class':<18}{'Supp':>6}  "
          f"{'Prec':>16}{'Rec':>16}{'F1':>16}  "
          f"{'Mean recon(xB)':>16}{'AE Det@τ(%)':>14}  "
          f"{'ROC-AUC':>10}{'Hybrid(%)':>12}")
    print("-"*110)

    for fam in FAMILIES:
        r = final.get(fam,{})
        if not r: continue
        label = fam if fam != "Benign" else "Benign"
        ae_col = r.get("ae_detection","—")
        if fam == "Benign":
            ae_col = f"{ae_col} (AE FPR)"
        hyb = r.get("hybrid_rate","—")
        if fam == "Benign":
            hyb = f"{hyb} (sys FPR)"
        print(f"{label:<18}{r.get('support',0):>6}  "
              f"{r.get('lgbm_precision','—'):>16}"
              f"{r.get('lgbm_recall','—'):>16}"
              f"{r.get('lgbm_f1','—'):>16}  "
              f"{r.get('ae_mean_err_x','—'):>16}"
              f"{ae_col:>14}  "
              f"{r.get('roc_auc','—'):>10}"
              f"{hyb:>12}")

    print("-"*110)
    print(f"{'Macro average':<18}{'—':>6}  "
          f"{'—':>16}"
          f"{macro.get('lgbm_recall','—'):>16}"
          f"{macro.get('lgbm_f1','—'):>16}  "
          f"{'--------':>16}"
          f"{macro.get('ae_detection','—'):>14}  "
          f"{macro.get('roc_auc','—'):>10}"
          f"{macro.get('hybrid_rate','—'):>12}")
    print("="*110)
    print(f"\nOverall accuracy (LightGBM): {mean_std(all_acc)}%")
    print(f"Macro F1 (LightGBM)        : {mean_std(all_f1)}%")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    jp = os.path.join(REPORT_DIR, f"known_attack_{ts}.json")
    mp = os.path.join(REPORT_DIR, f"known_attack_{ts}.md")

    with open(jp,"w") as f:
        json.dump({"timestamp":datetime.datetime.utcnow().isoformat(),
                   "seeds":SEEDS,"per_family":final,"macro":macro,
                   "overall_accuracy":mean_std(all_acc),
                   "overall_macro_f1":mean_std(all_f1)},f,indent=2)

    # Markdown table
    md  = "# Table 5: Known-Attack Detection Performance\n\n"
    md += f"**Generated:** {ts} UTC | **Seeds:** {SEEDS}\n\n"
    md += ("| Class | Supp | Prec | Rec | F1 | Mean recon (×benign) | "
           "AE Det @ τ (%) | ROC-AUC | Hybrid (%) |\n")
    md += "|---|---|---|---|---|---|---|---|---|\n"
    for fam in FAMILIES:
        r = final.get(fam,{})
        if not r: continue
        ae_col = r.get("ae_detection","—")
        if fam=="Benign": ae_col+=" (AE FPR)"
        hyb = r.get("hybrid_rate","—")
        if fam=="Benign": hyb+=" (sys FPR)"
        md += (f"| {fam} | {r.get('support',0)} | "
               f"{r.get('lgbm_precision','—')} | "
               f"{r.get('lgbm_recall','—')} | "
               f"{r.get('lgbm_f1','—')} | "
               f"{r.get('ae_mean_err_x','—')} | "
               f"{ae_col} | "
               f"{r.get('roc_auc','—')} | "
               f"{hyb} |\n")
    md += (f"| **Macro average** | — | — | "
           f"{macro.get('lgbm_recall','—')} | "
           f"{macro.get('lgbm_f1','—')} | -------- | "
           f"{macro.get('ae_detection','—')} | "
           f"{macro.get('roc_auc','—')} | "
           f"{macro.get('hybrid_rate','—')} |\n\n")
    md += f"**Overall accuracy (LightGBM):** {mean_std(all_acc)}%  \n"
    md += f"**Macro F1 (LightGBM):** {mean_std(all_f1)}%\n"

    with open(mp,"w") as f: f.write(md)
    print(f"\n[Eval] Saved:\n  JSON → {jp}\n  MD   → {mp}")


if __name__ == "__main__":
    run_evaluation()
