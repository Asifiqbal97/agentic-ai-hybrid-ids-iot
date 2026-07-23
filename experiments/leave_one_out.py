# =============================================================================
# experiments/leave_one_out.py — Unified LOO + Cross-dataset experiment
# Runs:
#   1. Leave-one-out (mean±std over N seeds) — fills Table zero-day rows
#   2. Cross-dataset Mirai/Benign (CICIoT2023) — fills cross-dataset rows
# Usage:
#   python experiments/leave_one_out.py            # both
#   python experiments/leave_one_out.py --loo-only # LOO only
#   python experiments/leave_one_out.py --cross-only # cross only
# =============================================================================

import os, sys, json, numpy as np, datetime, requests, argparse
from collections import Counter
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORT_DIR, LLM_HOST, LLM_TIMEOUT

try:
    import lightgbm as lgb
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import (roc_auc_score, roc_curve, recall_score,
                                  f1_score, accuracy_score, classification_report,
                                  precision_score)
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd, glob
except ImportError as e:
    print(f"Missing: {e}"); sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
N_SEEDS   = 3
SEEDS     = [42, 123, 999]
AE_EPOCHS = 20
LLM_MODEL = "qwen3:4b"

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
ATTACK_FAMILIES  = ["DDoS","DoS","Reconnaissance","MQTT","Spoofing"]
ZERODAY_PATTERNS = ["zero-day","zero day","unknown","anomaly","novel",
                     "unseen","unclassified","no signature","suspicious"]


# ── Autoencoder ───────────────────────────────────────────────────────────────
class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj  = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,
                                          dim_feedforward=128,batch_first=True)
        self.encoder     = nn.TransformerEncoder(enc,num_layers=num_layers)
        dec = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,
                                          dim_feedforward=128,batch_first=True)
        self.decoder     = nn.TransformerEncoder(dec,num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x=x.unsqueeze(1); x=self.input_proj(x)
        x=self.encoder(x); x=self.decoder(x)
        return self.output_proj(x).squeeze(1)


# ── Data helpers ──────────────────────────────────────────────────────────────
def _norm(label):
    for s in ["_train.pcap","_test.pcap","_train","_test",".pcap"]:
        if label.endswith(s): label=label[:-len(s)]; break
    return "Benign" if "benign" in label.lower() else label


def load_iomt():
    base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    frames=[]
    for folder,split in [
        (os.path.join(base,"data","raw","train_csvs"),"train"),
        (os.path.join(base,"data","raw","test_csvs"), "test")
    ]:
        for path in sorted(glob.glob(os.path.join(folder,"*.csv"))):
            label=_norm(os.path.splitext(os.path.basename(path))[0])
            df=pd.read_csv(path,low_memory=False)
            df.columns=df.columns.str.strip()
            df["label"]=label; df["split"]=split
            df["family"]=FAMILY_MAP.get(label,"Other")
            df.replace([np.inf,-np.inf],np.nan,inplace=True)
            df.dropna(inplace=True)
            frames.append(df)
    data=pd.concat(frames,ignore_index=True)
    print(f"[LOO] IoMT: {len(data):,} flows")
    return data


def load_cross():
    base=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cdir=os.path.join(base,"data","raw","cross_dataset")
    if not os.path.exists(cdir):
        print(f"[LOO] cross_dataset/ not found at {cdir}"); return None,None
    files=glob.glob(os.path.join(cdir,"*.csv"))
    if not files: print("[LOO] No CSVs in cross_dataset/"); return None,None
    frames=[]
    for path in files:
        fname=os.path.splitext(os.path.basename(path))[0].lower()
        df=pd.read_csv(path,low_memory=False)
        df.columns=df.columns.str.strip()
        df.replace([np.inf,-np.inf],np.nan,inplace=True)
        df.dropna(inplace=True)
        df["cross_label"]=fname
        frames.append(df)
    data=pd.concat(frames,ignore_index=True)
    mirai =data[data["cross_label"].str.contains("mirai", na=False)]
    benign=data[data["cross_label"].str.contains("benign",na=False)]
    print(f"[LOO] Cross: Mirai={len(mirai):,} Benign={len(benign):,}")
    return mirai,benign


# ── Model training ────────────────────────────────────────────────────────────
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
            b=b.to(device); l=loss_fn(model(b),b)
            opt.zero_grad(); l.backward(); opt.step()
        if (ep+1)%5==0: print(f"    AE ep {ep+1}/{epochs}")
    # Threshold on held-out benign val split
    X_tr2,X_val=train_test_split(X_normal,test_size=0.2,random_state=seed)
    model.eval()
    X_v=torch.tensor(X_val,dtype=torch.float32).to(device)
    with torch.no_grad():
        errs=torch.mean((model(X_v)-X_v)**2,dim=1).cpu().numpy()
    return model,float(np.percentile(errs,95)),device


def get_errors(model,X,device):
    model.eval()
    X_t=torch.tensor(X,dtype=torch.float32).to(device)
    with torch.no_grad():
        return torch.mean((model(X_t)-X_t)**2,dim=1).cpu().numpy()


# ── LLM flagging ──────────────────────────────────────────────────────────────
def llm_flag(lgbm_label,score,thr):
    p=(f"/no_think\nIDS: unknown_anomaly, lgbm={lgbm_label}, "
       f"score={score:.4f}, thr={thr:.4f}. Zero-day? yes/no briefly.")
    try:
        r=requests.post(f"{LLM_HOST}/api/chat",timeout=LLM_TIMEOUT,
            json={"model":LLM_MODEL,"messages":[{"role":"user","content":p}],
                  "stream":False,"options":{"num_ctx":128,"num_predict":32,"temperature":0.1}})
        a=r.json().get("message",{}).get("content","").lower()
        if "</think>" in a: a=a.split("</think>")[-1].strip()
        return any(x in a for x in ZERODAY_PATTERNS)
    except: return False


def llm_rate(scores,thr,labels,n=20):
    idxs=np.where(scores>thr)[0]
    if not len(idxs): return 0.0
    s=np.random.choice(idxs,min(n,len(idxs)),replace=False)
    flagged=sum(llm_flag(labels[i] if i<len(labels) else "benign",
                         float(scores[i]),thr) for i in s)
    rate=flagged/len(s)*100
    print(f"    LLM: {flagged}/{len(s)} = {rate:.1f}%")
    return round(rate,1)


# ── Mean±std helper ───────────────────────────────────────────────────────────
def ms(vals): return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"


# ── LOO single seed ───────────────────────────────────────────────────────────
def loo_seed(data,fcols,excluded,seed):
    np.random.seed(seed)
    tr=data[(data["split"]=="train")&(data["family"]!=excluded)]
    zd=data[(data["split"]=="test") &(data["family"]==excluded)]
    bn=data[(data["split"]=="test") &(data["family"]=="Benign")]
    if not len(zd): return None

    sc=MinMaxScaler()
    X_tr=sc.fit_transform(tr[fcols])
    X_zd=sc.transform(zd[fcols])
    X_bn=sc.transform(bn[fcols]) if len(bn) else np.zeros((0,len(fcols)))

    le=LabelEncoder(); le.fit(tr["label"])
    y_tr=le.transform(tr["label"])
    lgbm=train_lgbm(X_tr,y_tr,seed)
    preds=lgbm.predict(X_zd); lbls=le.inverse_transform(preds)
    bidx=le.transform(["Benign"])[0] if "Benign" in le.classes_ else -1
    dom=Counter(lbls).most_common(1)[0][0]

    X_norm=X_tr[(tr["label"]=="Benign").values]
    ae,thr,dev=train_ae(X_norm,seed)

    sc_zd=get_errors(ae,X_zd,dev); fl_zd=sc_zd>thr
    sc_bn=get_errors(ae,X_bn,dev) if len(X_bn) else np.array([])
    fl_bn=sc_bn>thr if len(sc_bn) else np.array([])

    ae_det =float(fl_zd.sum())/len(fl_zd)*100
    fpr    =float(fl_bn.sum())/len(fl_bn)*100 if len(fl_bn) else 0.0
    hybrid =((preds!=bidx)|fl_zd)
    hyb_rec=float(hybrid.sum())/len(hybrid)*100

    # ROC AUC
    roc_auc=0.0; fp_pts=[]; tp_pts=[]
    if len(sc_bn):
        at=np.concatenate([np.ones(len(sc_zd)),np.zeros(len(sc_bn))])
        as_=np.concatenate([sc_zd,sc_bn])
        if len(np.unique(at))>1:
            roc_auc=roc_auc_score(at,as_)
            fp,tp,_=roc_curve(at,as_)
            fp_pts,tp_pts=fp.tolist(),tp.tolist()

    llm_r=llm_rate(sc_zd,thr,lbls,n=20)

    return {"ae_detection_rate":round(ae_det,2),"hybrid_recall":round(hyb_rec,2),
            "benign_fpr":round(fpr,2),"dominant_lgbm":dom,"llm_flagging":llm_r,
            "roc_auc":round(roc_auc,4),"roc_fpr":fp_pts,"roc_tpr":tp_pts,
            "threshold":round(thr,6),"zeroday_flows":len(zd)}


# ── Cross-dataset single run ──────────────────────────────────────────────────
def cross_eval(data,fcols,mirai_df,benign_df,seed=42):
    """
    Train on full IoMT train data.
    Test on CICIoT2023 Mirai (zero-day) and Benign (FPR).
    Uses ONLY common features between IoMT and CICIoT2023.
    """
    np.random.seed(seed)
    tr=data[data["split"]=="train"]

    # Common features
    ccols=[c for c in fcols if c in mirai_df.columns]
    print(f"  Common features: {len(ccols)}/{len(fcols)}")
    if not ccols: return None

    sc=MinMaxScaler()
    X_tr=sc.fit_transform(tr[ccols])
    le=LabelEncoder(); le.fit(tr["label"])
    lgbm=train_lgbm(X_tr,le.transform(tr["label"]),seed)
    bidx=le.transform(["Benign"])[0] if "Benign" in le.classes_ else -1

    X_norm=X_tr[(tr["label"]=="Benign").values]
    ae,thr,dev=train_ae(X_norm,seed,epochs=15)

    out={}

    if len(mirai_df):
        X_m=sc.transform(mirai_df[ccols])
        sc_m=get_errors(ae,X_m,dev); fl_m=sc_m>thr
        prd_m=lgbm.predict(X_m); lbl_m=le.inverse_transform(prd_m)
        hyb_m=(prd_m!=bidx)|fl_m
        dom_m=Counter(lbl_m).most_common(1)[0][0]
        llm_m=llm_rate(sc_m,thr,lbl_m,n=20)
        out["mirai"]={"flows":len(mirai_df),
                      "ae_detection":round(float(fl_m.sum())/len(fl_m)*100,2),
                      "hybrid_recall":round(float(hyb_m.sum())/len(hyb_m)*100,2),
                      "dominant_lgbm":dom_m,"llm_flagging":llm_m}

    if benign_df is not None and len(benign_df):
        X_b=sc.transform(benign_df[ccols])
        fl_b=get_errors(ae,X_b,dev)>thr
        out["benign"]={"flows":len(benign_df),
                       "fpr":round(float(fl_b.sum())/len(fl_b)*100,2)}
    return out


# ── LOO full experiment ───────────────────────────────────────────────────────
def run_loo(data,fcols):
    print("\n[LOO] Leave-one-out experiment...")
    loo={}
    for fam in ATTACK_FAMILIES:
        print(f"\n{'='*55}\n[LOO] Excluded: {fam}\n{'='*55}")
        seed_res=[r for seed in SEEDS
                  if (r:=loo_seed(data,fcols,fam,seed)) is not None]
        if not seed_res: continue
        dom=Counter(r["dominant_lgbm"] for r in seed_res).most_common(1)[0][0]
        loo[fam]={
            "hybrid_recall":     ms([r["hybrid_recall"]     for r in seed_res]),
            "ae_detection_rate": ms([r["ae_detection_rate"] for r in seed_res]),
            "benign_fpr":        ms([r["benign_fpr"]        for r in seed_res]),
            "dominant_lgbm":     dom,
            "llm_flagging":      ms([r["llm_flagging"]      for r in seed_res]),
            "roc_auc":           ms([r["roc_auc"]           for r in seed_res]),
            "zeroday_flows":     seed_res[0]["zeroday_flows"],
            "roc_fpr":           seed_res[0].get("roc_fpr",[]),
            "roc_tpr":           seed_res[0].get("roc_tpr",[]),
        }
        for k,v in loo[fam].items():
            if k not in ["roc_fpr","roc_tpr"]:
                print(f"  {k:<22}: {v}")

    # Macro
    macro={}
    for key in ["hybrid_recall","ae_detection_rate","benign_fpr","llm_flagging"]:
        vals=[float(loo[f][key].split("±")[0]) for f in loo]
        macro[key]=round(np.mean(vals),2) if vals else 0.0

    return loo, macro


# ── Cross-dataset full experiment ─────────────────────────────────────────────
def run_cross(data,fcols,mirai_df,benign_df):
    print("\n[LOO] Cross-dataset experiment (CICIoT2023)...")
    seed_res=[]
    for seed in SEEDS:
        print(f"  Seed {seed}...")
        r=cross_eval(data,fcols,mirai_df,benign_df,seed)
        if r: seed_res.append(r)
    if not seed_res: return None

    out={}
    if "mirai" in seed_res[0]:
        out["mirai"]={
            "flows":         seed_res[0]["mirai"]["flows"],
            "ae_detection":  ms([r["mirai"]["ae_detection"]  for r in seed_res]),
            "hybrid_recall": ms([r["mirai"]["hybrid_recall"] for r in seed_res]),
            "dominant_lgbm": Counter(r["mirai"]["dominant_lgbm"]
                                     for r in seed_res).most_common(1)[0][0],
            "llm_flagging":  ms([r["mirai"]["llm_flagging"]  for r in seed_res]),
        }
    if "benign" in seed_res[0]:
        out["benign"]={
            "flows": seed_res[0]["benign"]["flows"],
            "fpr":   ms([r["benign"]["fpr"] for r in seed_res]),
        }
    return out


# ── Print & save ──────────────────────────────────────────────────────────────
def print_table(loo,macro,cross):
    print("\n\n"+"="*90)
    print("ZERO-DAY DETECTION TABLE (mean ± std)")
    print("="*90)
    print(f"{'Family':<18}{'Hybrid Recall':>18}{'AE Detection':>16}"
          f"{'Benign FPR':>12}{'Dominant LGBM':>18}{'LLM Flag%':>12}")
    print("-"*90)
    for fam in ATTACK_FAMILIES:
        r=loo.get(fam,{})
        if not r: continue
        print(f"{fam:<18}{r['hybrid_recall']:>18}{r['ae_detection_rate']:>16}"
              f"{r['benign_fpr']:>12}{r['dominant_lgbm']:>18}{r['llm_flagging']:>12}")
    print("-"*90)
    print(f"{'Macro average':<18}"
          f"{str(macro.get('hybrid_recall',0)):>18}"
          f"{str(macro.get('ae_detection_rate',0)):>16}"
          f"{str(macro.get('benign_fpr',0)):>12}"
          f"{'—':>18}"
          f"{str(macro.get('llm_flagging',0)):>12}")
    print("="*90)
    if cross:
        print("\nCross-Dataset (CICIoT2023):")
        m=cross.get("mirai",{}); b=cross.get("benign",{})
        if m: print(f"  Mirai  : Hybrid={m.get('hybrid_recall','—')}% "
                    f"AE={m.get('ae_detection','—')}% "
                    f"DOM={m.get('dominant_lgbm','—')} "
                    f"LLM={m.get('llm_flagging','—')}%")
        if b: print(f"  Benign : FPR={b.get('fpr','—')}%")


def save_results(loo,macro,cross):
    os.makedirs(REPORT_DIR,exist_ok=True)
    ts=datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    jp=os.path.join(REPORT_DIR,f"loo_unified_{ts}.json")
    mp=os.path.join(REPORT_DIR,f"loo_unified_{ts}.md")

    with open(jp,"w") as f:
        json.dump({"timestamp":datetime.datetime.utcnow().isoformat(),
                   "seeds":SEEDS,
                   "loo":{k:{kk:vv for kk,vv in v.items()
                              if kk not in ["roc_fpr","roc_tpr"]}
                          for k,v in loo.items()},
                   "macro":macro,"cross":cross},f,indent=2)

    md ="# Zero-Day Detection Results\n\n"
    md+=f"**Generated:** {ts} | **Seeds:** {SEEDS}\n\n"
    md+="## LOO Results (mean ± std)\n\n"
    md+=("| Family | Hybrid Recall | AE Detection | Benign FPR | "
         "Dominant LGBM | LLM Flagging |\n|---|---|---|---|---|---|\n")
    for fam in ATTACK_FAMILIES:
        r=loo.get(fam,{})
        if not r: continue
        md+=(f"| {fam} | {r['hybrid_recall']}% | {r['ae_detection_rate']}% | "
             f"{r['benign_fpr']}% | {r['dominant_lgbm']} | {r['llm_flagging']}% |\n")
    md+=(f"| **Macro** | **{macro.get('hybrid_recall',0)}%** | "
         f"**{macro.get('ae_detection_rate',0)}%** | "
         f"**{macro.get('benign_fpr',0)}%** | — | "
         f"**{macro.get('llm_flagging',0)}%** |\n\n")
    if cross:
        md+="## Cross-Dataset (CICIoT2023)\n\n"
        md+=("| Row | Hybrid | AE Detection | Benign FPR | "
             "Dominant LGBM | LLM Flagging |\n|---|---|---|---|---|---|\n")
        m=cross.get("mirai",{}); b=cross.get("benign",{})
        if m: md+=(f"| Mirai (CICIoT2023) | {m.get('hybrid_recall','—')}% | "
                   f"{m.get('ae_detection','—')}% | — | "
                   f"{m.get('dominant_lgbm','—')} | {m.get('llm_flagging','—')}% |\n")
        if b: md+=f"| Benign (CICIoT2023) | [FPR] | — | {b.get('fpr','—')}% | — | — |\n"

    with open(mp,"w") as f: f.write(md)
    print(f"\n[LOO] Saved:\n  JSON → {jp}\n  MD   → {mp}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--loo-only",   action="store_true")
    parser.add_argument("--cross-only", action="store_true")
    args=parser.parse_args()

    print("\n"+"="*65)
    print("UNIFIED LOO + CROSS-DATASET EXPERIMENT")
    print(f"Seeds={SEEDS}")
    print("="*65)

    data=load_iomt()
    fcols=[c for c in data.columns
           if c not in ["label","split","family"]
           and data[c].dtype in [np.float64,np.float32,np.int64,np.int32]]
    print(f"[LOO] Features: {len(fcols)}")

    loo={}; macro={}; cross=None

    run_l = not args.cross_only
    run_c = not args.loo_only

    if run_l:
        loo,macro=run_loo(data,fcols)

    if run_c:
        mirai_df,benign_df=load_cross()
        if mirai_df is not None:
            cross=run_cross(data,fcols,mirai_df,benign_df)

    print_table(loo,macro,cross)
    save_results(loo,macro,cross)
    print("\n[LOO] Done.")


if __name__=="__main__":
    main()
