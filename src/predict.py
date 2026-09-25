"""Predict whether a single article is FAKE or TRUE with the saved model.

Usage:
    python src/predict.py --text "Article text here..."
    python src/predict.py --title "Headline" --text "Article text..."
    python src/predict.py --file article.txt
    python src/predict.py                  # paste the article, then Ctrl-D
                                           # (Windows: Ctrl-Z, then Enter)

The saved pipeline applies the same preprocessing and TF-IDF vocabulary used
in training; nothing is re-fitted.
"""

import argparse
import sys
from pathlib import Path

from utils import CLASS_NAMES, load_model, print_header


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
    label = int(model.predict([article])[0])
    score = float(model.decision_function([article])[0])
    known_terms = model.named_steps["tfidf"].transform([article]).nnz
    return CLASS_NAMES[label], score, known_terms


def main():
    parser = argparse.ArgumentParser(description="Classify a news article as FAKE or TRUE.")
    parser.add_argument("--text", help="article text")
    parser.add_argument("--title", help="optional headline (combined with the text)")
    parser.add_argument("--file", help="path to a .txt file containing the article")
    args = parser.parse_args()

    article = read_article(args)
    if not article:
        sys.exit("No article text provided.")

    model = load_model()
    label, score, known_terms = predict_article(model, article)

    print("\n")
    print(f"Prediction:     {label}")
    print(f"Decision score: {score:+.3f}  (positive = FAKE, negative = TRUE; "
          f"further from 0 = more confident)")
    
    if known_terms < 20:
        print("Warning: very little of this text matches the training vocabulary, "
              "so the prediction is unreliable.")
    


if __name__ == "__main__":
    main()
