"""
ASRRL vs Black-Box Model Comparison

Compares the full Adaptive Symbolic Reasoning + Reinforcement Learning (ASRRL)
framework against black-box baselines:
  - Random Forest
  - XGBoost
  - LightGBM

Generates:
  1. Side-by-side performance (Acc, Precision, Recall, F1) per dataset
  2. FPR / FNR comparison
  3. Radar chart (multi-metric profile)
  4. Training time comparison
  5. Per-dataset detailed table (printed + saved as CSV)

Usage:
    python compare_models.py              # 5000 samples
    python compare_models.py 20000        # more samples
"""

import sys, os, time, copy, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Import ASRRL framework from experiments.py
from experiments import (
    run_framework, _generate, _metrics, _fp_fn,
    FEATURE_NAMES, DATASETS, DATASET_LABELS
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.15)

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ORDER = ["ASRRL\n(Ours)", "Random\nForest", "XGBoost", "LightGBM"]
MODEL_COLORS = {
    "ASRRL\n(Ours)":  "#2980b9",
    "Random\nForest":  "#27ae60",
    "XGBoost":         "#e67e22",
    "LightGBM":        "#8e44ad",
}


# ═══════════════════════════════════════════════════════════════════════════
#  RUN ALL MODELS
# ═══════════════════════════════════════════════════════════════════════════

def run_all(n, epochs=10):
    """Return dict[ds][model_name] = {y_true, y_pred, train_time, ...}"""
    black_box_models = {
        "Random\nForest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0),
        "LightGBM": LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            random_state=42, verbose=-1),
    }

    all_results = {}

    for ds in DATASETS:
        all_results[ds] = {}
        df = _generate(n, ds)
        split = int(len(df) * 0.7)
        train_df, test_df = df.iloc[:split], df.iloc[split:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[FEATURE_NAMES])
        y_train = train_df["label"].values.astype(int)
        X_test = scaler.transform(test_df[FEATURE_NAMES])
        y_test = test_df["label"].values.astype(int)

        # ── ASRRL (our framework) ────────────────────────────────────────
        t0 = time.time()
        res = run_framework(n, ds, epochs=epochs)
        asrrl_time = time.time() - t0
        all_results[ds]["ASRRL\n(Ours)"] = {
            "y_true": res.y_true, "y_pred": res.y_pred,
            "train_time": asrrl_time,
            "shields": res.shield_activations,
            "z3_constraints": res.constraint_log[-1][1] if res.constraint_log else 0,
            "novel_patterns": res.n_novel_patterns,
        }
        m = _metrics(res.y_true, res.y_pred)
        fpr, fnr = _fp_fn(res.y_true, res.y_pred)
        print(f"  {DATASET_LABELS[ds]:20s}  ASRRL (Ours)    "
              f"Acc={m['Accuracy']:.3f}  P={m['Precision']:.3f}  "
              f"R={m['Recall']:.3f}  F1={m['F1']:.3f}  "
              f"FPR={fpr:.3f}  FNR={fnr:.3f}  t={asrrl_time:.1f}s")

        # ── Black-box baselines ───────────────────────────────────────────
        for name, clf_template in black_box_models.items():
            clf = copy.deepcopy(clf_template)
            t0 = time.time()
            clf.fit(X_train, y_train)
            preds = clf.predict(X_test)
            bb_time = time.time() - t0

            all_results[ds][name] = {
                "y_true": y_test, "y_pred": preds,
                "train_time": bb_time,
            }
            m = _metrics(y_test, preds)
            fpr, fnr = _fp_fn(y_test, preds)
            print(f"  {DATASET_LABELS[ds]:20s}  {name:15s} "
                  f"Acc={m['Accuracy']:.3f}  P={m['Precision']:.3f}  "
                  f"R={m['Recall']:.3f}  F1={m['F1']:.3f}  "
                  f"FPR={fpr:.3f}  FNR={fnr:.3f}  t={bb_time:.1f}s")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 1: Performance bar chart (Acc / P / R / F1)
# ═══════════════════════════════════════════════════════════════════════════

def fig_performance(all_results):
    rows = []
    for ds in DATASETS:
        for model in MODEL_ORDER:
            r = all_results[ds][model]
            m = _metrics(r["y_true"], r["y_pred"])
            for metric, val in m.items():
                rows.append({"Dataset": DATASET_LABELS[ds], "Model": model,
                             "Metric": metric, "Value": val})
    mdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        sub = mdf[mdf["Dataset"] == DATASET_LABELS[ds]]
        sns.barplot(data=sub, x="Metric", y="Value", hue="Model",
                    hue_order=MODEL_ORDER, palette=MODEL_COLORS,
                    edgecolor="black", linewidth=0.5, ax=ax)
        ax.set_title(DATASET_LABELS[ds], fontweight="bold", fontsize=12)
        ax.set_ylim(0, 1.12)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=7, padding=2, rotation=90)
        ax.legend(title="Model", fontsize=8, title_fontsize=9)
    fig.suptitle("ASRRL (Ours) vs Black-Box Baselines — Performance Comparison",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "cmp_1_performance.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 2: FPR / FNR comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig_fp_fn(all_results):
    rows = []
    for ds in DATASETS:
        for model in MODEL_ORDER:
            r = all_results[ds][model]
            fpr, fnr = _fp_fn(r["y_true"], r["y_pred"])
            rows.append({"Dataset": DATASET_LABELS[ds], "Model": model,
                         "Error": "False Positive Rate", "Value": fpr})
            rows.append({"Dataset": DATASET_LABELS[ds], "Model": model,
                         "Error": "False Negative Rate", "Value": fnr})
    rdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, err in zip(axes, ["False Positive Rate", "False Negative Rate"]):
        sub = rdf[rdf["Error"] == err]
        sns.barplot(data=sub, x="Dataset", y="Value", hue="Model",
                    hue_order=MODEL_ORDER, palette=MODEL_COLORS,
                    edgecolor="black", linewidth=0.5, ax=ax)
        ax.set_title(err, fontweight="bold", fontsize=12)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
        ax.legend(fontsize=8)
    fig.suptitle("ASRRL vs Black-Box — Error Rate Analysis",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "cmp_2_fp_fn.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 3: Radar chart (multi-metric profile)
# ═══════════════════════════════════════════════════════════════════════════

def fig_radar(all_results):
    metrics_list = ["Accuracy", "Precision", "Recall", "F1", "1-FPR", "1-FNR"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                              subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, len(metrics_list), endpoint=False).tolist()
    angles += angles[:1]

    for ax, ds in zip(axes, DATASETS):
        ax.set_title(DATASET_LABELS[ds], fontweight="bold", fontsize=11, pad=20)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics_list, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)

        for model in MODEL_ORDER:
            r = all_results[ds][model]
            m = _metrics(r["y_true"], r["y_pred"])
            fpr, fnr = _fp_fn(r["y_true"], r["y_pred"])
            vals = [m["Accuracy"], m["Precision"], m["Recall"], m["F1"],
                    1 - fpr, 1 - fnr]
            vals += vals[:1]
            lw = 2.5 if "ASRRL" in model else 1.2
            ls = "-" if "ASRRL" in model else "--"
            ax.plot(angles, vals, linewidth=lw, linestyle=ls,
                    label=model.replace("\n", " "),
                    color=MODEL_COLORS[model])
            ax.fill(angles, vals, alpha=0.08, color=MODEL_COLORS[model])
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.suptitle("Multi-Metric Radar — ASRRL vs Black-Box",
                 fontweight="bold", fontsize=14, y=1.02)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "cmp_3_radar.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 4: Training time comparison
# ═══════════════════════════════════════════════════════════════════════════

def fig_train_time(all_results):
    rows = []
    for ds in DATASETS:
        for model in MODEL_ORDER:
            t = all_results[ds][model]["train_time"]
            rows.append({"Dataset": DATASET_LABELS[ds], "Model": model,
                         "Time (s)": t})
    tdf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=tdf, x="Dataset", y="Time (s)", hue="Model",
                hue_order=MODEL_ORDER, palette=MODEL_COLORS,
                edgecolor="black", linewidth=0.5, ax=ax)
    ax.set_title("Training + Inference Time Comparison",
                 fontweight="bold", fontsize=13)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8, padding=2)
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "cmp_4_train_time.png")
    fig.savefig(p, dpi=200); plt.close(fig)
    print(f"  Saved {p}")


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE 5: Comprehensive comparison heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig_heatmap(all_results):
    rows = []
    for ds in DATASETS:
        for model in MODEL_ORDER:
            r = all_results[ds][model]
            m = _metrics(r["y_true"], r["y_pred"])
            fpr, fnr = _fp_fn(r["y_true"], r["y_pred"])
            rows.append({
                "Dataset": DATASET_LABELS[ds],
                "Model": model.replace("\n", " "),
                "Accuracy": m["Accuracy"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1": m["F1"],
                "FPR": fpr,
                "FNR": fnr,
            })
    full_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(20, 4.5))
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"]
    for ax, ds in zip(axes, DATASETS):
        sub = full_df[full_df["Dataset"] == DATASET_LABELS[ds]].set_index("Model")
        heat_data = sub[metrics_cols].astype(float)
        sns.heatmap(heat_data, annot=True, fmt=".3f", cmap="RdYlGn",
                    linewidths=0.8, ax=ax, vmin=0, vmax=1,
                    cbar_kws={"shrink": 0.8})
        ax.set_title(DATASET_LABELS[ds], fontweight="bold", fontsize=11)
        ax.set_ylabel("")
    fig.suptitle("ASRRL vs Black-Box — Full Metric Heatmap",
                 fontweight="bold", fontsize=14, y=1.03)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "cmp_5_heatmap.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved {p}")

    # Also save CSV
    csv_path = os.path.join(OUT_DIR, "comparison_table.csv")
    full_df.to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")
    return full_df


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    epochs = 10
    print(f"ASRRL vs Black-Box Comparison  (n={n}, epochs={epochs})\n")

    all_results = run_all(n, epochs)

    print(f"\nGenerating comparison figures in ./{OUT_DIR}/ ...\n")
    fig_performance(all_results)
    fig_fp_fn(all_results)
    fig_radar(all_results)
    fig_train_time(all_results)
    table_df = fig_heatmap(all_results)

    print("\n" + "=" * 80)
    print("  COMPARISON TABLE")
    print("=" * 80)
    print(table_df.to_string(index=False))
    print("=" * 80)

    # Highlight ASRRL advantages
    print("\n  KEY FINDINGS:")
    for ds in DATASETS:
        asrrl = all_results[ds]["ASRRL\n(Ours)"]
        asrrl_f1 = f1_score(asrrl["y_true"], asrrl["y_pred"], zero_division=0)
        best_bb_name, best_bb_f1 = None, 0
        for model in ["Random\nForest", "XGBoost", "LightGBM"]:
            r = all_results[ds][model]
            f = f1_score(r["y_true"], r["y_pred"], zero_division=0)
            if f > best_bb_f1:
                best_bb_f1 = f
                best_bb_name = model.replace("\n", " ")
        diff = asrrl_f1 - best_bb_f1
        sign = "+" if diff >= 0 else ""
        print(f"    {DATASET_LABELS[ds]:20s}  ASRRL F1={asrrl_f1:.3f}  "
              f"vs best black-box ({best_bb_name}) F1={best_bb_f1:.3f}  "
              f"  delta={sign}{diff:.3f}"
              f"  | Z3 constraints={asrrl.get('z3_constraints', 'N/A')}"
              f"  shields={asrrl.get('shields', 'N/A')}")

    print(f"\n  ASRRL advantage: interpretable (Z3 constraints extractable as")
    print(f"  human-readable rules), safety-shielded (provably safe actions),")
    print(f"  and adaptive (DBSCAN detects novel attack patterns).")
    print(f"\n  All figures saved to {OUT_DIR}/\n")


if __name__ == "__main__":
    main()
