# experiments/plot_roc.py
import json, glob, os
import matplotlib.pyplot as plt

# Load latest LOO result
files = sorted(glob.glob("reports/loo_unified_*.json"))
if not files:
    print("No LOO results found. Run leave_one_out.py first.")
    exit()

data    = json.load(open(files[-1]))
loo     = data.get("loo", {})
families= ["DDoS","DoS","Reconnaissance","MQTT","Spoofing"]

plt.figure(figsize=(8,6))
for fam in families:
    r = loo.get(fam, {})
    fpr = r.get("roc_fpr", [])
    tpr = r.get("roc_tpr", [])
    auc = r.get("roc_auc", "N/A")
    if fpr and tpr:
        plt.plot(fpr, tpr, label=f"{fam} (AUC={auc})")

plt.plot([0,1],[0,1],"k--",label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — AE Anomaly Score per LOFO Fold")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/roc_curves.png", dpi=150)
plt.show()
print("Saved: reports/roc_curves.png")
