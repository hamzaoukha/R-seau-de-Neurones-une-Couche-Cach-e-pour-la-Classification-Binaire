import argparse, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from .data import load_and_prepare
from .nn import NeuralNetwork
from .threshold import class_weights_from, tune_threshold

def run(csv, mode="baseline", out_dir="reports", seed=42):
    Xtr, ytr, Xv, yv, Xte, yte = load_and_prepare(csv, seed=seed)

    out_dir = Path(out_dir)
    (out_dir/"figures").mkdir(parents=True, exist_ok=True)
    (out_dir/"metrics").mkdir(parents=True, exist_ok=True)

    if mode == "baseline":
        net = NeuralNetwork([8,16,8,1], learning_rate=0.01, l2=0.0, use_adam=False, class_weights=None, seed=seed)
        trL, vaL = net.train(Xtr, ytr, Xv, yv, epochs=100, batch_size=32, seed=seed)
        thr = 0.5
        p_te = net.predict_proba(Xte); yhat = (p_te > thr).astype(int)

        # plots
        plt.figure(figsize=(6.2,3.3))
        plt.plot(trL, label="Train"); plt.plot(vaL, label="Val")
        plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.title("Baseline learning curves"); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir/"figures"/"baseline_curves.png", dpi=300); plt.close()

        cm = confusion_matrix(yte, yhat)
        ConfusionMatrixDisplay(cm).plot(cmap="Blues", colorbar=False)
        plt.title("Baseline confusion matrix (test)")
        plt.tight_layout(); plt.savefig(out_dir/"figures"/"baseline_cm.png", dpi=300); plt.close()

        acc, pre, rec, f1 = [accuracy_score(yte,yhat), precision_score(yte,yhat), recall_score(yte,yhat), f1_score(yte,yhat)]
        print("\n=== Baseline (test) ===\n", classification_report(yte, yhat))
        json.dump({"mode":"baseline","threshold":thr,"acc":acc,"prec":pre,"rec":rec,"f1":f1},
                  open(out_dir/"metrics"/"baseline.json","w"), indent=2)

    elif mode == "improved":
        cw = class_weights_from(ytr)
        net = NeuralNetwork([8,32,16,1], learning_rate=0.001, l2=1e-3, use_adam=True, class_weights=cw, seed=seed)
        trL, vaL = net.train(Xtr, ytr, Xv, yv, epochs=120, batch_size=32, seed=seed)
        thr, best_f1 = tune_threshold(net.predict_proba(Xv), yv)

        # test
        p_te = net.predict_proba(Xte); yhat = (p_te > thr).astype(int)

        # plots
        plt.figure(figsize=(6.2,3.3))
        plt.plot(trL, label="Train"); plt.plot(vaL, label="Val")
        plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.title("Improved learning curves"); plt.legend()
        plt.tight_layout(); plt.savefig(out_dir/"figures"/"improved_curves.png", dpi=300); plt.close()

        cm = confusion_matrix(yte, yhat)
        ConfusionMatrixDisplay(cm).plot(cmap="Greens", colorbar=False)
        plt.title("Improved confusion matrix (test)")
        plt.tight_layout(); plt.savefig(out_dir/"figures"/"improved_cm.png", dpi=300); plt.close()

        acc, pre, rec, f1 = [accuracy_score(yte,yhat), precision_score(yte,yhat), recall_score(yte,yhat), f1_score(yte,yhat)]
        print(f"Best validation threshold τ={thr:.2f} (val F1={best_f1:.3f})")
        print("\n=== Improved (test) ===\n", classification_report(yte, yhat))
        json.dump({"mode":"improved","threshold":float(thr),"acc":acc,"prec":pre,"rec":rec,"f1":f1},
                  open(out_dir/"metrics"/"improved.json","w"), indent=2)
    else:
        raise ValueError("mode must be 'baseline' or 'improved'")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to diabetes.csv")
    ap.add_argument("--mode", choices=["baseline","improved"], default="baseline")
    ap.add_argument("--out_dir", default="reports")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.csv, mode=args.mode, out_dir=args.out_dir, seed=args.seed)
