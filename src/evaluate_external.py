"""Experiment 2: saved WELFake-trained pipelines -> unseen PolyglotFakeFacts.

No fit() / fit_transform() is called on PolyglotFakeFacts. Each saved pipeline
only applies its stored preprocessing, TF-IDF vocabulary and classifier.
"""

from utils import (
    MODEL_NAMES,
    RESULTS_DIR,
    evaluate_model,
    load_metrics,
    load_model,
    load_polyglot,
    metrics_table,
    plot_confusion_matrix,
    print_header,
    print_metrics,
    save_metrics,
)


def evaluate_external():
    X, y = load_polyglot()
    results = {}
    for name in MODEL_NAMES:
        model = load_model(name)
        vocab_size_before = len(model.named_steps["tfidf"].vocabulary_)

        y_pred = model.predict(X)  # tfidf.transform() + classifier.predict() only

        # Guard: the vocabulary must be exactly the one learned from WELFake.
        assert len(model.named_steps["tfidf"].vocabulary_) == vocab_size_before

        metrics = evaluate_model(y, y_pred)
        plot_confusion_matrix(metrics["confusion_matrix"],
                              f"PolyglotFakeFacts External — {MODEL_NAMES[name]}",
                              RESULTS_DIR / f"polyglot_external_confusion_matrix_{name}.png")
        save_metrics("experiment_2_polyglot_external", name, metrics)
        results[name] = metrics
    return results


def main():
    results = evaluate_external()

    for name, metrics in results.items():
        print_header("EXPERIMENT 2 — POLYGLOTFAKEFACTS EXTERNAL TEST\n"
                     f"(Cross-Dataset External Evaluation) — {MODEL_NAMES[name].upper()}")
        print_metrics(metrics)

    exp1 = load_metrics().get("experiment_1_welfake_validation", {})
    print_header("MODEL AND CROSS-DATASET COMPARISON")
    columns = {}
    for name, metrics in results.items():
        if name in exp1:
            columns[f"{MODEL_NAMES[name]} — WELFake"] = exp1[name]
        columns[f"{MODEL_NAMES[name]} — Polyglot"] = metrics
    if not exp1:
        print("Experiment 1 metrics not found. Run `python src/train.py` first.\n")
    print(metrics_table(columns))


if __name__ == "__main__":
    main()
