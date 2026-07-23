# =============================================================================
# experiments/leave_one_out.py — Cross-dataset only mode
# Runs ONLY the cross-dataset evaluation (CICIoT2023)
# Usage: python experiments/leave_one_out.py
# =============================================================================

import os, sys, json, numpy as np, datetime, requests
from collections import Counter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, REPORT_DIR, LLM_HOST, LLM_TIMEOUT

try:
    import lightgbm as lgb
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    import pandas as pd
    import glob
except ImportError as e:
    print(f"Missing: {e}"); sys.exit(1)

LLM_MODEL = "qwen3:4b"
AE_EPOCHS = 20

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
    "ARP_Spoofing":            "Spoofing",
    "Benign":                  "Benign",
}

ZERODAY_PATTERNS = ["zero-day","zero day","unknown","anomaly","novel",
                     "unseen","unclassified","no signature","suspicious"]


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
def _normalize_label(label):
    for s in ["_train.pcap","_test.pcap","_train","_test",".pcap"]:
        if label.endswith(s):
            label = label[:-len(s)]; break
    return "Benign" if "benign" in label.lower() else label


def load_iomt_data():
    base      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TRAIN_DIR = os.path.join(base,"data","raw","train_csvs")
    frames    = []
    for path in sorted(glob.glob(os.path.join(TRAIN_DIR,"*.csv"))):
        label = _normalize_label(os.path.splitext(os.path.basename(path))[0])
        df    = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        df["label"]  = label
        df["family"] = FAMILY_MAP.get(label,"Other")
        df.replace([np.inf,-np.inf],np.nan,inplace=True)
        df.dropna(inplace=True)
        frames.append(df)
    data = pd.concat(frames,ignore_index=True)
    print(f"[Cross] IoMT train data: {len(data):,} flows")
    return data


def load_cross_dataset():
    """Load CICIoT2023 CSVs — uses filename as label."""
    base      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cross_dir = os.path.join(base,"data","raw","cross_dataset")

    if not os.path.exists(cross_dir):
        print(f"[Cross] Folder not found: {cross_dir}"); return None, None

    csv_files = sorted(glob.glob(os.path.join(cross_dir,"*.csv")))
    if not csv_files:
        print("[Cross] No CSVs found"); return None, None

    print(f"[Cross] Found {len(csv_files)} CSV files:")
    frames = []
    for path in csv_files:
        filename = os.path.splitext(os.path.basename(path))[0].lower()
        print(f"  {filename}")
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        df.replace([np.inf,-np.inf],np.nan,inplace=True)
        df.dropna(inplace=True)
        df["cross_label"] = filename
        frames.append(df)

    data   = pd.concat(frames,ignore_index=True)
    mirai  = data[data["cross_label"].str.contains("mirai",  na=False)]
    benign = data[data["cross_label"].str.contains("benign", na=False)]

    print(f"[Cross] Mirai flows : {len(mirai):,}")
    print(f"[Cross] Benign flows: {len(benign):,}")
    return mirai, benign


# ── Training helpers ──────────────────────────────────────────────────────────
def train_lgbm(X, y, seed=42):
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05,
                             num_leaves=63, random_state=seed,
                             n_jobs=-1, verbose=-1)
    m.fit(X,y); return m


def train_ae(X_normal, seed=42, epochs=AE_EPOCHS):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = TransformerAutoencoder(X_normal.shape[1]).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn= nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_normal, dtype=torch.float32)),
        batch_size=256, shuffle=True
    )
    model.train()
    for ep in range(epochs):
        for (b,) in loader:
            b = b.to(device)
            l = loss_fn(model(b), b)
            opt.zero_grad(); l.backward(); opt.step()
        if (ep+1) % 5 == 0:
            print(f"  AE epoch {ep+1}/{epochs}")
    model.eval()
    X_t = torch.tensor(X_normal, dtype=torch.float32).to(device)
    with torch.no_grad():
        errs = torch.mean((model(X_t)-X_t)**2, dim=1).cpu().numpy()
    return model, float(np.percentile(errs,95)), device


def ae_scores(model, X, device):
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        return torch.mean((model(X_t)-X_t)**2, dim=1).cpu().numpy()


# ── LLM flagging ──────────────────────────────────────────────────────────────
def llm_flag(lgbm_label, anomaly_score, threshold) -> bool:
    prompt = (f"/no_think\nIDS: attack=unknown_anomaly, lgbm={lgbm_label}, "
              f"score={anomaly_score:.4f}, threshold={threshold:.4f}. "
              f"Zero-day? Answer yes/no briefly.")
    try:
        r = requests.post(f"{LLM_HOST}/api/chat", timeout=LLM_TIMEOUT,
            json={"model":LLM_MODEL,
                  "messages":[{"role":"user","content":prompt}],
                  "stream":False,
                  "options":{"num_ctx":128,"num_predict":32,"temperature":0.1}})
        ans = r.json().get("message",{}).get("content","").lower()
        if "</think>" in ans:
            ans = ans.split("</think>")[-1].strip()
        return any(p in ans for p in ZERODAY_PATTERNS)
    except:
        return False


def compute_llm_flagging(scores, thr, lgbm_labels, n=20) -> float:
    idxs = np.where(scores > thr)[0]
    if len(idxs) == 0: return 0.0
    sample  = np.random.choice(idxs, min(n,len(idxs)), replace=False)
    flagged = sum(llm_flag(
        lgbm_labels[i] if i < len(lgbm_labels) else "benign",
        float(scores[i]), thr
    ) for i in sample)
    rate = flagged / len(sample) * 100
    print(f"  LLM flagging: {flagged}/{len(sample)} = {rate:.1f}%")
    return round(rate, 1)


# ── Cross-dataset evaluation ──────────────────────────────────────────────────
def run_cross(iomt_data, feature_cols, mirai_df, benign_df):
    print("\n[Cross] Training models on IoMT 2024 data...")

    # Common features between IoMT and CICIoT2023
    common_cols = [c for c in feature_cols if c in mirai_df.columns]
    print(f"[Cross] Common features: {len(common_cols)}/{len(feature_cols)}")

    if len(common_cols) == 0:
        print("[Cross] No common features — cannot run cross-dataset")
        print(f"[Cross] IoMT features  : {feature_cols[:5]}...")
        print(f"[Cross] Cross features : {list(mirai_df.columns[:5])}...")
        return None

    # Scale using IoMT training data
    sc     = MinMaxScaler()
    X_tr   = sc.fit_transform(iomt_data[common_cols])

    # Encode IoMT labels
    le     = LabelEncoder()
    le.fit(iomt_data["label"])
    y_tr   = le.transform(iomt_data["label"])
    bidx   = le.transform(["Benign"])[0] if "Benign" in le.classes_ else -1

    # Train LightGBM on full IoMT training data
    print("[Cross] Training LightGBM...")
    lgbm   = train_lgbm(X_tr, y_tr)

    # Train AE on benign IoMT flows only
    X_norm = X_tr[(iomt_data["label"]=="Benign").values]
    print(f"[Cross] Training AE on {len(X_norm):,} benign flows...")
    ae, thr, dev = train_ae(X_norm, seed=42, epochs=AE_EPOCHS)
    print(f"[Cross] AE threshold: {thr:.6f}")

    out = {}

    # ── Mirai flows ───────────────────────────────────────────────────────────
    if len(mirai_df) > 0:
        print(f"\n[Cross] Evaluating Mirai ({len(mirai_df):,} flows)...")
        X_m      = sc.transform(mirai_df[common_cols])
        sc_m     = ae_scores(ae, X_m, dev)
        fl_m     = sc_m > thr
        prd_m    = lgbm.predict(X_m)
        lbl_m    = le.inverse_transform(prd_m)
        hyb_m    = (prd_m != bidx) | fl_m
        dom_m    = Counter(lbl_m).most_common(1)[0][0]
        ae_det_m = round(float(fl_m.sum())/len(fl_m)*100, 2)
        hyb_m_r  = round(float(hyb_m.sum())/len(hyb_m)*100, 2)
        llm_m    = compute_llm_flagging(sc_m, thr, lbl_m, n=20)

        out["mirai"] = {
            "flows":          len(mirai_df),
            "ae_detection":   ae_det_m,
            "hybrid_recall":  hyb_m_r,
            "dominant_lgbm":  dom_m,
            "llm_flagging":   llm_m,
            "mean_ae_score":  round(float(sc_m.mean()), 6),
            "anomaly_ratio":  round(float(sc_m.mean())/thr, 2),
        }
        print(f"  AE detection  : {ae_det_m:.1f}%")
        print(f"  Hybrid recall : {hyb_m_r:.1f}%")
        print(f"  Dominant LGBM : {dom_m}")
        print(f"  Anomaly ratio : {out['mirai']['anomaly_ratio']:.2f}x threshold")

    # ── Benign flows (FPR only) ───────────────────────────────────────────────
    if benign_df is not None and len(benign_df) > 0:
        print(f"\n[Cross] Evaluating Benign ({len(benign_df):,} flows)...")
        X_b  = sc.transform(benign_df[common_cols])
        sc_b = ae_scores(ae, X_b, dev)
        fl_b = sc_b > thr
        fpr  = round(float(fl_b.sum())/len(fl_b)*100, 2)
        out["benign"] = {"flows": len(benign_df), "fpr": fpr}
        print(f"  Benign FPR: {fpr:.1f}%")

    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n"+"="*65)
    print("CROSS-DATASET EVALUATION (CICIoT2023)")
    print("="*65)

    # Load IoMT training data
    iomt_data = load_iomt_data()

    # Load cross-dataset
    mirai_df, benign_df = load_cross_dataset()
    if mirai_df is None and benign_df is None:
        print("[Cross] No cross-dataset data available. Exiting.")
        return

    # Feature columns from IoMT
    feature_cols = [c for c in iomt_data.columns
                    if c not in ["label","family"]
                    and iomt_data[c].dtype in [np.float64,np.float32,
                                               np.int64,np.int32]]
    print(f"[Cross] IoMT features : {len(feature_cols)}")

    # Run cross-dataset evaluation
    results = run_cross(iomt_data, feature_cols, mirai_df, benign_df)

    if not results:
        print("[Cross] No results generated.")
        return

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n\n"+"="*75)
    print("CROSS-DATASET RESULTS (CICIoT2023)")
    print("="*75)
    print(f"{'Row':<22}{'Hybrid Recall':>14}{'AE Detection':>14}"
          f"{'Benign FPR':>12}{'Dominant LGBM':>16}{'LLM Flag%':>10}")
    print("-"*75)

    m = results.get("mirai",{})
    b = results.get("benign",{})

    if m:
        print(f"{'Mirai (CICIoT2023)':<22}"
              f"{str(m.get('hybrid_recall',0))+'%':>14}"
              f"{str(m.get('ae_detection',0))+'%':>14}"
              f"{'—':>12}"
              f"{m.get('dominant_lgbm','—'):>16}"
              f"{str(m.get('llm_flagging',0))+'%':>10}")
    if b:
        print(f"{'Benign (CICIoT2023)':<22}"
              f"{'[FPR]':>14}"
              f"{'—':>14}"
              f"{str(b.get('fpr',0))+'%':>12}"
              f"{'—':>16}"
              f"{'—':>10}")
    print("="*75)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    jp = os.path.join(REPORT_DIR, f"cross_dataset_{ts}.json")
    mp = os.path.join(REPORT_DIR, f"cross_dataset_{ts}.md")

    with open(jp,"w") as f:
        json.dump({"timestamp": datetime.datetime.utcnow().isoformat(),
                   "results": results}, f, indent=2)

    md  = "# Cross-Dataset Evaluation Results (CICIoT2023)\n\n"
    md += f"**Generated:** {ts} UTC\n\n"
    md += ("| Row | Hybrid detection rate | AE-only detection rate | "
           "Benign FPR | Dominant LightGBM misattribution | LLM side zero-day flagging |\n")
    md += "|---|---|---|---|---|---|\n"
    if m:
        md += (f"| Mirai (CICIoT2023) | {m.get('hybrid_recall',0):.1f}% | "
               f"{m.get('ae_detection',0):.1f}% | — | "
               f"{m.get('dominant_lgbm','—')} | {m.get('llm_flagging',0):.1f}% |\n")
    if b:
        md += f"| Benign (CICIoT2023) | [FPR] | — | {b.get('fpr',0):.1f}% | — | — |\n"

    with open(mp,"w") as f: f.write(md)

    print(f"\n[Cross] Saved:")
    print(f"  JSON → {jp}")
    print(f"  MD   → {mp}")


if __name__ == "__main__":
    main()
