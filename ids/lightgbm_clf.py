# =============================================================================
# ids/lightgbm_clf.py — Train and run LightGBM for known attack classification
# =============================================================================

import os
import pickle
import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LGBM_MODEL_PATH, LGBM_CONFIDENCE_THRESH


def train(X_train, y_train, X_test, y_test, label_encoder):
    """Train LightGBM and save model to disk."""
    print("[LightGBM] Training...")

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("[LightGBM] Classification Report:")
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    target_names  = label_encoder.classes_[unique_labels]
    print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))

    # Save
    with open(LGBM_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"[LightGBM] Model saved → {LGBM_MODEL_PATH}")

    return model


def load_model():
    """Load trained LightGBM model from disk."""
    with open(LGBM_MODEL_PATH, "rb") as f:
        return pickle.load(f)


def predict(model, features: np.ndarray, label_encoder) -> dict:
    """
    Run inference on a single feature vector.
    Returns: { label, confidence, is_attack }
    """
    proba   = model.predict_proba([features])[0]
    idx     = int(np.argmax(proba))
    label   = label_encoder.classes_[idx]
    confidence = float(proba[idx])

    return {
        "lgbm_label":      label,
        "lgbm_confidence": round(confidence, 4),
        "lgbm_is_attack":  label.lower() != "benign" and confidence >= LGBM_CONFIDENCE_THRESH
    }


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.preprocess import preprocess

    X_train, X_test, y_train, y_test, le, scaler = preprocess()
    model = train(X_train, y_train, X_test, y_test, le)
