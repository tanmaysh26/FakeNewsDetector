"""Experiment 2: saved WELFake-trained pipeline -> unseen PolyglotFakeFacts.

No fit() / fit_transform() is called on PolyglotFakeFacts. The saved pipeline
only applies its stored preprocessing, TF-IDF vocabulary and SVM weights.
"""

from utils import (
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
    model = load_model()
    vocab_size_before = len(model.named_steps["tfidf"].vocabulary_)

    X, y = load_polyglot()
    y_pred = model.predict(X)  # tfidf.transform() + svm.predict() only

    # Guard: the vocabulary must be exactly the one learned from WELFake.
    assert len(model.named_steps["tfidf"].vocabulary_) == vocab_size_before

    metrics = evaluate_model(y, y_pred)
    plot_confusion_matrix(metrics["confusion_matrix"],
                          "PolyglotFakeFacts — Cross-Dataset External Evaluation",
                          RESULTS_DIR / "polyglot_external_confusion_matrix.png")
    save_metrics("experiment_2_polyglot_external", metrics)
    return metrics


def main():
    metrics = evaluate_external()

    print_header("EXPERIMENT 2 — POLYGLOTFAKEFACTS EXTERNAL TEST\n"
                 "(Cross-Dataset External Evaluation)")
    print_metrics(metrics)

    exp1 = load_metrics().get("experiment_1_welfake_validation")
    print_header("CROSS-DATASET COMPARISON")
    if exp1 is None:
        print("Experiment 1 metrics not found. Run `python src/train.py` first.")
    else:
        print(metrics_table({"WELFake Validation": exp1,
                             "PolyglotFakeFacts External Test": metrics}))


if __name__ == "__main__":
    main()
