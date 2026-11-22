"""VLDBench experiment: TF-IDF + Linear SVM trained and tested on VLDBench,
text only (headline + article text, text_label).

VLDBench has no official train/test split (one Hugging Face split, "train"),
so a reproducible 80/20 stratified split is used, as for WELFake. TF-IDF is
fitted on the training part only. Results go to results/vldbench/ and the
model to models/vldbench/, so earlier WELFake/Polyglot outputs are untouched.
"""

import csv
import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils import (
    MODELS_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    build_svm,
    build_vectorizer,
    evaluate_model,
    load_vldbench,
    plot_confusion_matrix,
    print_header,
    print_metrics,
)

OUT_DIR = RESULTS_DIR / "vldbench"
MODEL_PATH = MODELS_DIR / "vldbench" / "fake_news_tfidf_svm_vldbench.joblib"

# Published text-only results, copied from the VLDBench paper (Raza et al.,
# arXiv:2502.11361v5; Information Fusion 2026). These are NOT produced by this
# project. Table 9: zero-shot prompting of language-only LLMs, whole dataset,
# mean over three runs, values in %. The paper does not state the averaging
# used for precision/recall/F1; the official eval script (code/eval291124.py)
# uses average="weighted".
PAPER_SOURCE = "Raza et al., VLDBench, arXiv:2502.11361v5"
PAPER_ZERO_SHOT = [
    # model, accuracy, precision, recall, f1
    ("Phi-3-mini-128k-Instruct", 57.15, 55.12, 58.21, 56.62),
    ("Vicuna-7B-v1.5", 55.21, 56.78, 52.48, 54.55),
    ("Mistral-7B-Instruct-v0.3", 68.58, 68.10, 65.09, 66.56),
    ("Qwen2-7B-Instruct", 69.92, 68.34, 69.10, 68.72),
    ("InternLM2-7B", 51.19, 49.85, 51.16, 50.50),
    ("DeepSeek-V2-Lite-Chat", 51.96, 52.61, 53.53, 51.96),
    ("GLM-4-9B-Chat", 51.14, 60.19, 53.28, 56.52),
    ("LLaMA-3.1-8B-Instruct", 68.21, 62.13, 62.10, 62.11),
    ("LLaMA-3.2-1B-Instruct", 70.29, 69.78, 68.92, 69.35),
]
# Section 5.2 / Figure 9: the only text-only instruction-tuned (IFT) number
# given in the text. Only F1 is reported; it is measured on the paper's own
# 15% test split (70/15/15 by outlet and date), which is not released.
PAPER_IFT = [("Phi-3-mini-128k-Instruct (IFT)", None, None, None, 67.2)]


def averaged_scores(y_true, y_pred):
    """Precision/recall/F1 with weighted and macro averaging over both classes."""
    out = {}
    for avg in ("weighted", "macro"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        out[avg] = {"precision": p, "recall": r, "f1": f}
    return out


def summary(y_true, y_pred):
    """The numbers used in the comparison table, in %."""
    w = averaged_scores(y_true, y_pred)["weighted"]
    return [100 * accuracy_score(y_true, y_pred), 100 * w["precision"],
            100 * w["recall"], 100 * w["f1"]]


def build_model():
    """Same architecture and settings as the WELFake experiment (C=1.0)."""
    return Pipeline([("tfidf", build_vectorizer()), ("svm", build_svm())])


def evaluate(model, X, y):
    y_pred = model.predict(X)
    metrics = evaluate_model(y, y_pred)  # binary, FAKE (= Likely) positive
    metrics["averaged"] = averaged_scores(y, y_pred)
    return metrics, y_pred


def outlet_majority_predictions(train_df, test_df):
    """Reference baseline: predict each outlet's most common training label
    (overall training majority for outlets not seen in training)."""
    overall = int(train_df["label"].mode()[0])
    per_outlet = train_df.groupby("outlet")["label"].agg(lambda s: int(s.mode()[0]))
    return test_df["outlet"].map(per_outlet).fillna(overall).astype(int).to_numpy()


def save_comparison(rows):
    header = ["model", "result_type", "setting", "evaluation_data",
              "accuracy", "precision", "recall", "f1", "source"]
    with open(OUT_DIR / "comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{v:.2f}" if isinstance(v, float) else ("" if v is None else v)
                             for v in row])

    fmt = lambda v: "n/a" if v is None else f"{v:.2f}"
    lines = ["| Model | Result type | Setting | Evaluation data | Accuracy | Precision | Recall | F1 | Source |",
             "|---|---|---|---|---:|---:|---:|---:|---|"]
    for model, kind, setting, data, *scores, source in rows:
        lines.append(f"| {model} | {kind} | {setting} | {data} | "
                     + " | ".join(fmt(s) for s in scores) + f" | {source} |")
    (OUT_DIR / "comparison.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_vldbench()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    print(f"[Split] train: {len(train_df)} | test: {len(test_df)} "
          f"(80/20 stratified, random_state={RANDOM_STATE})")
    seen = test_df["outlet"].isin(set(train_df["outlet"])).mean()
    print(f"[Split] test articles whose outlet also appears in training: {seen:.2%}")

    X_train, y_train = train_df["text"], train_df["label"]
    X_test, y_test = test_df["text"], test_df["label"].to_numpy()

    # Pipeline.fit fits TF-IDF on the training texts only; predict() only transforms.
    model = build_model().fit(X_train, y_train)
    print(f"[TF-IDF] vocabulary size (fitted on training split only): "
          f"{len(model.named_steps['tfidf'].vocabulary_)}")
    metrics, y_pred = evaluate(model, X_test, y_test)

    majority = np.full(len(y_test), int(train_df["label"].mode()[0]))
    by_outlet = outlet_majority_predictions(train_df, test_df)
    baselines = {
        "majority_class": {"accuracy": accuracy_score(y_test, majority),
                           **averaged_scores(y_test, majority)},
        "outlet_majority": {"accuracy": accuracy_score(y_test, by_outlet),
                            **averaged_scores(y_test, by_outlet)},
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    test_df[["unique_id"]].to_csv(OUT_DIR / "test_ids.csv", index=False)
    plot_confusion_matrix(metrics["confusion_matrix"],
                          "VLDBench Test (text only) — Linear SVM",
                          OUT_DIR / "confusion_matrix.png")
    (OUT_DIR / "classification_report.txt").write_text(metrics["classification_report"])
    (OUT_DIR / "metrics.json").write_text(json.dumps({
        "dataset": "vector-institute/VLDBench (text_label, headline + article_text)",
        "positive_class": "FAKE = 'Likely' disinformation",
        "split": {"method": "80/20 stratified random split", "random_state": RANDOM_STATE,
                  "n_train": int(len(train_df)), "n_test": int(len(test_df)),
                  "test_outlets_seen_in_training": float(seen)},
        "linear_svm": {"C": 1.0, **metrics},
        "baselines": baselines,
    }, indent=2))

    test_data = f"VLDBench, our 20% test split (n={len(test_df)})"
    rows = [
        ("TF-IDF + Linear SVM (C=1.0)", "this experiment", "supervised, trained on our 80% split",
         test_data, *summary(y_test, y_pred), "results/vldbench/metrics.json"),
        ("Majority class (always Likely)", "this experiment (reference)", "no learning",
         test_data, *summary(y_test, majority), "results/vldbench/metrics.json"),
        ("Outlet majority label", "this experiment (reference)", "outlet name only",
         test_data, *summary(y_test, by_outlet), "results/vldbench/metrics.json"),
    ]
    rows += [(m, "published", "zero-shot prompting, text only", "VLDBench, whole dataset",
              *scores, f"{PAPER_SOURCE}, Table 9") for m, *scores in PAPER_ZERO_SHOT]
    rows += [(m, "published", "instruction fine-tuned, text only",
              "VLDBench, paper's own 15% test split", *scores,
              f"{PAPER_SOURCE}, Section 5.2 / Figure 9") for m, *scores in PAPER_IFT]
    table = save_comparison(rows)

    print_header("VLDBENCH TEST — TF-IDF + LINEAR SVM, C=1.0 (TEXT ONLY)\n"
                 "Binary metrics, positive class = FAKE ('Likely')")
    print_metrics(metrics)
    for avg, s in metrics["averaged"].items():
        print(f"{avg:>8} avg: precision {s['precision']:.4f} | recall {s['recall']:.4f} | f1 {s['f1']:.4f}")
    for name, b in baselines.items():
        print(f"[Baseline] {name}: accuracy {b['accuracy']:.4f} | "
              f"weighted F1 {b['weighted']['f1']:.4f} | macro F1 {b['macro']['f1']:.4f}")

    print_header("COMPARISON WITH PUBLISHED TEXT-ONLY RESULTS (weighted averages, %)")
    print(table)
    print(f"\nSaved results to {OUT_DIR.relative_to(RESULTS_DIR.parent)}/ "
          f"and the model to {MODEL_PATH.relative_to(MODELS_DIR.parent)}")


if __name__ == "__main__":
    main()
