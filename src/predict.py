"""Classify a single article as FAKE or TRUE with every saved model.

Usage:
    python src/predict.py --text "Article text here..."
    python src/predict.py --title "Headline" --text "Article text..."
    python src/predict.py --file article.txt
    python src/predict.py                  # paste the article, then Ctrl-D
                                           # (Windows: Ctrl-Z, then Enter)

Each saved pipeline applies the same preprocessing and TF-IDF vocabulary used
in training; nothing is re-fitted.
"""

import argparse
import sys
from pathlib import Path

from utils import CLASS_NAMES, LABEL_FAKE, MODEL_NAMES, classifier_step, load_model


def read_article(args):
    if args.file:
        body = Path(args.file).read_text(encoding="utf-8", errors="ignore")
    elif args.text:
        body = args.text
    else:
        if sys.stdin.isatty():
            print("Paste the article below. Press Ctrl-D when done "
                  "(Windows: Ctrl-Z, then Enter):\n")
        body = sys.stdin.read()
    # Same input format as training: title + " " + text.
    return f"{args.title or ''} {body}".strip()


def predict_article(model, article):
    """Returns (label, formatted score, what the score means, known terms)."""
    label = CLASS_NAMES[int(model.predict([article])[0])]
    known_terms = model.named_steps["tfidf"].transform([article]).nnz
    clf = classifier_step(model)
    if hasattr(clf, "predict_proba"):
        fake_col = list(clf.classes_).index(LABEL_FAKE)
        score = float(model.predict_proba([article])[0][fake_col])
        return (label, f"{score:.3f}",
                "P(FAKE): share of FAKE training articles in the leaf", known_terms)
    score = float(model.decision_function([article])[0])
    return (label, f"{score:+.3f}",
            "positive = FAKE, negative = TRUE; further from 0 = more confident", known_terms)


def main():
    parser = argparse.ArgumentParser(description="Classify a news article as FAKE or TRUE.")
    parser.add_argument("--text", help="article text")
    parser.add_argument("--title", help="optional headline (combined with the text)")
    parser.add_argument("--file", help="path to a .txt file containing the article")
    args = parser.parse_args()

    article = read_article(args)
    if not article:
        sys.exit("No article text provided.")

    print(f"\n{'Model':<15} {'Prediction':<11} {'Score':>7}   Meaning")
    for name in MODEL_NAMES:
        label, score, meaning, known_terms = predict_article(load_model(name), article)
        print(f"{MODEL_NAMES[name]:<15} {label:<11} {score:>7}   {meaning}")

    # Both models share one vocabulary, so the match count is the same.
    print(f"\nTerms matching the training vocabulary: {known_terms}")
    if known_terms < 20:
        print("Warning: very little of this text matches the training vocabulary, "
              "so the predictions are unreliable.")


if __name__ == "__main__":
    main()
