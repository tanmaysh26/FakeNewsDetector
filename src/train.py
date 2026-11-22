"""Experiment 1: WELFake -> 80/20 split -> TF-IDF -> Linear SVM, Decision
Tree, Logistic Regression and Naive Bayes -> validation.

One TF-IDF vectorizer is fitted on the WELFake training split only, and every
classifier is trained on its features, so the comparison is like-for-like.
Each trained pipeline is saved and reused unchanged for Experiment 2.
"""

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from utils import (
    MODEL_NAMES,
    MODEL_PATHS,
    RANDOM_STATE,
    RESULTS_DIR,
    build_classifiers,
    build_vectorizer,
    evaluate_model,
    load_welfake,
    metrics_table,
    plot_confusion_matrix,
    print_header,
    print_metrics,
    save_metrics,
    save_model,
)


def train_models(X_train, y_train):
    """Fit TF-IDF once on the training split, then each classifier on it.
    Returns {name: Pipeline(tfidf, classifier)}."""
    tfidf = build_vectorizer()
    features = tfidf.fit_transform(X_train)
    print(f"[TF-IDF] vocabulary size (fitted on training split only): "
          f"{len(tfidf.vocabulary_)}")
    models = {}
    for name, clf in build_classifiers().items():
        print(f"[Train] fitting {MODEL_NAMES[name]}...")
        clf.fit(features, y_train)
        models[name] = Pipeline([("tfidf", tfidf), (name, clf)])
    return models


def main():
    X, y = load_welfake()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[Split] train: {len(X_train)} | validation: {len(X_val)}")

    models = train_models(X_train, y_train)

    results = {}
    for name, model in models.items():
        # predict() only calls tfidf.transform() on the validation texts.
        metrics = evaluate_model(y_val, model.predict(X_val))
        results[name] = metrics
        plot_confusion_matrix(metrics["confusion_matrix"],
                              f"WELFake Validation — {MODEL_NAMES[name]}",
                              RESULTS_DIR / f"welfake_validation_confusion_matrix_{name}.png")
        save_model(model, name)
        save_metrics("experiment_1_welfake_validation", name, {
            **metrics, "n_train": int(len(X_train)), "n_validation": int(len(X_val))})

        print_header(f"EXPERIMENT 1 — WELFAKE VALIDATION — {MODEL_NAMES[name].upper()}")
        print_metrics(metrics)

    print_header("MODEL COMPARISON — WELFAKE VALIDATION")
    print(metrics_table({MODEL_NAMES[n]: m for n, m in results.items()}))
    for name, path in MODEL_PATHS.items():
        print(f"{MODEL_NAMES[name]} saved to: {path.relative_to(path.parents[1])}")


if __name__ == "__main__":
    main()
