"""Step 3 (demo): predict the composer of a single .mid file.

Usage:  python src/predict.py path/to/piece.midi
"""
import sys

import joblib
import numpy as np
import pandas as pd

from features import extract_features


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python src/predict.py path/to/file.mid")
    bundle = joblib.load("models/best_model.joblib")
    pipe, cols = bundle["pipeline"], bundle["features"]

    feats = extract_features(sys.argv[1])
    x = pd.DataFrame([feats])[cols]
    proba = pipe.predict_proba(x)[0]

    print(f"\nModel: {bundle['name']}")
    print(f"File:  {sys.argv[1]}\n")
    print("Predicted composer probabilities:")
    for i in np.argsort(proba)[::-1]:
        print(f"  {pipe.classes_[i]:35s} {proba[i]:6.1%}  {'#' * int(proba[i] * 40)}")

    # which features make this piece stand out from the training average (z-scores)
    z = pipe.named_steps["standardscaler"].transform(x)[0]
    print("\nMost unusual features of this piece (z-score vs. training average):")
    for i in np.argsort(-np.abs(z))[:5]:
        print(f"  {cols[i]:28s} z = {z[i]:+.2f}")


if __name__ == "__main__":
    main()
