"""Experiment 1: WELFake -> 80/20 split -> TF-IDF -> Linear SVM -> validation.

The pipeline trained here (TF-IDF fitted on the WELFake training split only)
is saved and reused unchanged for Experiment 2.
"""

from sklearn.model_selection import train_test_split

from utils import (
    MODEL_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    build_pipeline,
    evaluate_model,
    load_welfake,
    metrics_table,
    plot_confusion_matrix,
    print_header,
    print_metrics,
    save_metrics,
    save_model,
)


def train_model(X_train, y_train):
    """Pipeline.fit calls tfidf.fit_transform(X_train), then svm.fit(...)."""
    model = build_pipeline()
    model.fit(X_train, y_train)
    return model


def main():
    X, y = load_welfake()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[Split] train: {len(X_train)} | validation: {len(X_val)}")

    model = train_model(X_train, y_train)
    print(f"[TF-IDF] vocabulary size (fitted on training split only): "
          f"{len(model.named_steps['tfidf'].vocabulary_)}")

    # predict() only calls tfidf.transform() on the validation texts.
    y_pred = model.predict(X_val)
    metrics = evaluate_model(y_val, y_pred)

    plot_confusion_matrix(metrics["confusion_matrix"],
                          "WELFake Validation — Confusion Matrix",
                          RESULTS_DIR / "welfake_validation_confusion_matrix.png")
    save_model(model)
    save_metrics("experiment_1_welfake_validation", {
        **metrics, "n_train": int(len(X_train)), "n_validation": int(len(X_val))})

    print_header("EXPERIMENT 1 — WELFAKE VALIDATION")
    print_metrics(metrics)
    print(metrics_table({"WELFake Validation": metrics}))
    print(f"\nModel saved to: {MODEL_PATH.relative_to(MODEL_PATH.parents[1])}")


if __name__ == "__main__":
    main()
