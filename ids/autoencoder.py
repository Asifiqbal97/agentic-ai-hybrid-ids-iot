# =============================================================================
# ids/autoencoder.py — Transformer Autoencoder for zero-day anomaly detection
# Trained on normal (benign) traffic only.
# Flags a flow as anomalous if reconstruction error > learned threshold.
# =============================================================================

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (AE_MODEL_PATH, AE_THRESHOLD_PATH, AE_EPOCHS,
                    AE_BATCH_SIZE, AE_LEARNING_RATE, AE_ANOMALY_PERCENTILE)


# ── Model definition ──────────────────────────────────────────────────────────

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()

        # Encoder: project input → d_model, apply Transformer
        self.input_proj  = nn.Linear(input_dim, d_model)
        encoder_layer    = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                       dim_feedforward=128, batch_first=True)
        self.encoder     = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder: mirror of encoder
        decoder_layer    = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                       dim_feedforward=128, batch_first=True)
        self.decoder     = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        # x shape: (batch, input_dim) → add sequence dim → (batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_proj(x)
        return x.squeeze(1)


# ── Training ──────────────────────────────────────────────────────────────────

def train(X_normal: np.ndarray):
    """Train autoencoder on normal traffic. Save model + threshold."""
    print("[Autoencoder] Training on normal traffic...")

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim  = X_normal.shape[1]

    X_tensor   = torch.tensor(X_normal, dtype=torch.float32)
    loader     = DataLoader(TensorDataset(X_tensor), batch_size=AE_BATCH_SIZE, shuffle=True)

    model      = TransformerAutoencoder(input_dim).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=AE_LEARNING_RATE)
    criterion  = nn.MSELoss()

    model.train()
    for epoch in range(AE_EPOCHS):
        total_loss = 0
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            loss  = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{AE_EPOCHS} — loss: {total_loss/len(loader):.6f}")

    # ── Compute threshold from reconstruction errors on normal data ────────────
    model.eval()
    with torch.no_grad():
        recon   = model(X_tensor.to(device))
        errors  = torch.mean((recon - X_tensor.to(device)) ** 2, dim=1).cpu().numpy()

    threshold = float(np.percentile(errors, AE_ANOMALY_PERCENTILE))
    print(f"[Autoencoder] Threshold ({AE_ANOMALY_PERCENTILE}th percentile): {threshold:.6f}")

    # Save model and threshold
    torch.save(model.state_dict(), AE_MODEL_PATH)
    with open(AE_THRESHOLD_PATH, "wb") as f:
        pickle.dump({"threshold": threshold, "input_dim": input_dim}, f)
    print(f"[Autoencoder] Saved → {AE_MODEL_PATH}")

    return model, threshold


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model():
    """Load trained autoencoder and threshold from disk."""
    with open(AE_THRESHOLD_PATH, "rb") as f:
        meta = pickle.load(f)

    model = TransformerAutoencoder(meta["input_dim"])
    model.load_state_dict(torch.load(AE_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, meta["threshold"]


def predict(model, threshold: float, features: np.ndarray) -> dict:
    """
    Run inference on a single feature vector.
    Returns: { anomaly_score, threshold, is_anomaly }
    """
    x     = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        recon = model(x)
    error = float(torch.mean((recon - x) ** 2).item())

    return {
        "anomaly_score":     round(error, 6),
        "anomaly_threshold": round(threshold, 6),
        "is_anomaly":        error > threshold
    }


if __name__ == "__main__":
    from data.preprocess import preprocess, get_normal_data

    X_train, X_test, y_train, y_test, le, scaler = preprocess()
    X_normal = get_normal_data(X_train, y_train, le)
    model, threshold = train(X_normal)
    print("[Autoencoder] Done.")
