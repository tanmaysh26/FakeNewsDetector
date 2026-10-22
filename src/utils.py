"""Shared utilities for the TF-IDF + Linear SVM fake news project."""

import html
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
WELFAKE_DIR = ROOT / "data" / "welfake"
POLYGLOT_DIR = ROOT / "data" / "polyglot"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
MODEL_PATHS = {
    "svm": MODELS_DIR / "fake_news_tfidf_svm.joblib",
    "decision_tree": MODELS_DIR / "fake_news_tfidf_decision_tree.joblib",
}
MODEL_NAMES = {"svm": "Linear SVM", "decision_tree": "Decision Tree"}
METRICS_PATH = RESULTS_DIR / "metrics.json"

RANDOM_STATE = 42

# Common binary encoding used for both datasets. FAKE is the positive class,
# so precision/recall/F1 are reported for detecting fake news.
LABEL_TRUE = 0
LABEL_FAKE = 1
CLASS_NAMES = ["TRUE", "FAKE"]

# WELFake label meaning. The dataset documentation states "0 = fake and
# 1 = real", but the CSV itself shows the reverse, verified on WELFake only:
#   - 21,270 articles containing the Reuters dateline "(Reuters)" (real news)
#     have label 0, versus 19 with label 1;
#   - the class sizes (35,028 label-0 vs 37,106 label-1) match the documented
#     counts of 35,028 real and 37,106 fake articles.
# So in the actual file: 0 = real, 1 = fake.
WELFAKE_LABEL_MAP = {0: LABEL_TRUE, 1: LABEL_FAKE}

# PolyglotFakeFacts: the documentation lists these fields (exact header
# spelling is not documented, so columns are auto-detected and printed):
# gathering date, news date, language, URL, web domain, keywords,
# news headline, news original text, english translated version, label.
# Set any entry to the exact column name to override auto-detection.
POLYGLOT_COLUMNS = {
    "headline": None,
    "text": None,  # must be the ENGLISH translated text (WELFake is English)
    "label": None,
    "language": None,
}

# Only used if the Polyglot label column is not a recognisable fake/real
# string (e.g. numeric codes). Fill it in ONLY after checking the dataset's
# documentation, e.g. {0: LABEL_TRUE, 1: LABEL_FAKE}.
POLYGLOT_LABEL_MAP = None

_FAKE_STRINGS = {"fake", "fake news", "false", "disinformation"}
_TRUE_STRINGS = {"true", "real", "true news", "real news", "reliable",
                 "non-fake", "non fake", "non_fake", "nonfake"}


# ---------------------------------------------------------------------------
# Preprocessing (identical for both datasets; embedded in the saved pipeline)
# ---------------------------------------------------------------------------
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+")
_WHITESPACE = re.compile(r"\s+")


def prepare_text(text):
    """Light cleaning: HTML, URLs, lowercase, whitespace. No stemming/stopwords."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = html.unescape(str(text))
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.lower()
    return _WHITESPACE.sub(" ", text).strip()


def combine_title_text(title, text):
    """combined_text = title + " " + text, with missing values treated as ''."""
    title = title.fillna("").astype(str)
    text = text.fillna("").astype(str)
    return (title + " " + text).str.strip()


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------
def _norm(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _find_column(columns, exact, contains=(), exclude=(), required=True, what=""):
    normed = {_norm(c): c for c in columns}
    for cand in exact:
        if _norm(cand) in normed:
            return normed[_norm(cand)]
    for n, original in normed.items():
        if any(k in n for k in contains) and not any(e in n for e in exclude):
            return original
    if required:
        raise ValueError(
            f"Could not find the {what} column. Available columns: {list(columns)}. "
            f"Set it explicitly in utils.py."
        )
    return None


def _single_data_file(folder, patterns):
    files = sorted(f for p in patterns for f in folder.glob(p))
    if not files:
        raise FileNotFoundError(f"No data file ({', '.join(patterns)}) found in {folder}")
    return files


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------
def load_welfake():
    """Load WELFake, inspect its structure, return (texts, labels)."""
    files = _single_data_file(WELFAKE_DIR, ["*.csv"])
    path = files[0]
    df = pd.read_csv(path)
    print(f"[WELFake] file: {path.name} | rows: {len(df)} | columns: {list(df.columns)}")

    title_col = _find_column(df.columns, ["title"], what="WELFake title")
    text_col = _find_column(df.columns, ["text"], what="WELFake text")
    label_col = _find_column(df.columns, ["label"], what="WELFake label")
    print(f"[WELFake] using title='{title_col}', text='{text_col}', label='{label_col}'")
    print(f"[WELFake] missing values: title={df[title_col].isna().sum()}, "
          f"text={df[text_col].isna().sum()}, label={df[label_col].isna().sum()}")

    df = df.dropna(subset=[label_col])
    raw = pd.to_numeric(df[label_col], errors="coerce")
    unknown = set(raw.dropna().unique()) - set(WELFAKE_LABEL_MAP)
    if raw.isna().any() or unknown:
        raise ValueError(f"Unexpected WELFake label values: {df[label_col].unique()[:20]}")

    # Sanity check of the documented label meaning.
    for value in sorted(WELFAKE_LABEL_MAP):
        sample = df.loc[raw == value, title_col].dropna().head(2).tolist()
        print(f"[WELFake] raw label {value} -> {CLASS_NAMES[WELFAKE_LABEL_MAP[value]]}; "
              f"sample titles: {[s[:70] for s in sample]}")

    out = pd.DataFrame({
        "text": combine_title_text(df[title_col], df[text_col]),
        "label": raw.astype(int).map(WELFAKE_LABEL_MAP),
    })
    before = len(out)
    out = out[out["text"].str.len() > 0]
    # Exact duplicates would let identical articles land in both train and
    # validation, inflating Experiment 1 scores.
    out = out.drop_duplicates(subset="text")
    print(f"[WELFake] removed {before - len(out)} empty/duplicate rows | final: {len(out)} | "
          f"TRUE={int((out.label == LABEL_TRUE).sum())}, FAKE={int((out.label == LABEL_FAKE).sum())}")
    return out["text"].reset_index(drop=True), out["label"].to_numpy()


SOURCE_FILE_COL = "__source_file__"


def _label_from_filename(stem):
    """Class defined by the dataset's class files: Fake.xlsx / Non_Fake.xlsx."""
    n = _norm(stem)
    if "non_fake" in n or "nonfake" in n or n in {"real", "true"}:
        return "real"
    if "fake" in n:
        return "fake"
    return None


def _normalize_columns(df):
    """Lowercase/strip headers so files with different capitalisation
    ("Label" vs "label") line up. Same-named duplicates are merged."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip().lower() for c in df.columns]
    if df.columns.duplicated().any():
        merged = {}
        for name in dict.fromkeys(df.columns):
            block = df.loc[:, df.columns == name]
            merged[name] = block.bfill(axis=1).iloc[:, 0] if block.shape[1] > 1 else block.iloc[:, 0]
        df = pd.DataFrame(merged, index=df.index)
    return df


def _map_polyglot_labels(series):
    if POLYGLOT_LABEL_MAP is not None:
        mapped = series.map(POLYGLOT_LABEL_MAP)
    else:
        normed = series.astype(str).str.strip().str.lower()
        mapped = normed.map(
            lambda v: LABEL_FAKE if v in _FAKE_STRINGS else LABEL_TRUE if v in _TRUE_STRINGS else np.nan
        )
    if mapped.isna().any():
        bad = series[mapped.isna()].unique()[:20]
        raise ValueError(
            f"Unrecognised PolyglotFakeFacts label values: {list(bad)}. Check the dataset "
            f"documentation and set POLYGLOT_LABEL_MAP in utils.py."
        )
    return mapped.astype(int)


def load_polyglot():
    """Load PolyglotFakeFacts, inspect its structure, return (texts, labels)."""
    files = _single_data_file(POLYGLOT_DIR, ["*.xlsx", "*.xls", "*.csv"])
    frames = []
    for path in files:
        if path.suffix == ".csv":
            parts = [pd.read_csv(path)]
            print(f"[Polyglot] file: {path.name} | rows: {len(parts[0])}")
        else:
            sheets = pd.read_excel(path, sheet_name=None)
            parts = list(sheets.values())
            print(f"[Polyglot] file: {path.name} | sheets: {list(sheets)} | "
                  f"rows: {sum(len(x) for x in parts)}")
        for part in parts:
            part = _normalize_columns(part)
            part[SOURCE_FILE_COL] = path.stem
            frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    print(f"[Polyglot] rows: {len(df)} | columns: {list(df.columns)}")

    cols = dict(POLYGLOT_COLUMNS)
    label_col = cols["label"] or _find_column(
        df.columns, ["label", "labels", "class"], required=False)
    cols["text"] = cols["text"] or _find_column(
        df.columns, ["english_translated_version", "english_translation", "text_en"],
        contains=("english", "translat"), exclude=("headline", "title"),
        what="Polyglot English-translated text")
    cols["headline"] = cols["headline"] or (
        _find_column(df.columns, [], contains=("headline_en", "english_headline",
                                              "translated_headline"), required=False)
        or _find_column(df.columns, ["news_headline", "headline", "title"], required=False))
    cols["language"] = cols["language"] or _find_column(
        df.columns, ["language"], contains=("lang",), required=False)
    print(f"[Polyglot] using: {dict(cols, label=label_col)}")

    used = [c for c in [cols["headline"], cols["text"], label_col, cols["language"]] if c]
    print(f"[Polyglot] missing values: {df[used].isna().sum().to_dict()}")
    if cols["language"]:
        # Display only: some cells repeat the language on several lines.
        lang = df[cols["language"]].astype(str).str.split("\n").str[0].str.strip().str.title()
        print(f"[Polyglot] languages: {lang.value_counts().to_dict()}")

    # Labels: the explicit Label column where present, otherwise the class
    # defined by the file the row comes from (Fake / Non_Fake). Any row whose
    # explicit label contradicts its file stops the script.
    file_label = df[SOURCE_FILE_COL].map(_label_from_filename)
    explicit = df[label_col] if label_col else pd.Series(np.nan, index=df.index)
    for name, group in df.groupby(SOURCE_FILE_COL):
        counts = explicit[group.index].value_counts(dropna=False).to_dict()
        print(f"[Polyglot] {name}: file class={file_label[group.index].iloc[0]} | "
              f"label column values={counts}")
    both = explicit.notna() & file_label.notna()
    if both.any():
        conflicts = int((_map_polyglot_labels(explicit[both])
                         != _map_polyglot_labels(file_label[both])).sum())
        if conflicts:
            raise ValueError(f"{conflicts} rows have a label that contradicts their file.")
    filled = int((explicit.isna() & file_label.notna()).sum())
    print(f"[Polyglot] {filled} rows with an empty label column labeled from their file")

    df = df.assign(__label__=explicit.where(explicit.notna(), file_label))
    df = df.dropna(subset=["__label__"])
    headline = df[cols["headline"]] if cols["headline"] else pd.Series("", index=df.index)
    out = pd.DataFrame({
        "text": combine_title_text(headline, df[cols["text"]]),
        "label": _map_polyglot_labels(df["__label__"]),
    })
    out = out[out["text"].str.len() > 0]
    n_true = int((out.label == LABEL_TRUE).sum())
    n_fake = int((out.label == LABEL_FAKE).sum())
    print(f"[Polyglot] final: {len(out)} | TRUE={n_true}, FAKE={n_fake}")
    if n_true == 0 or n_fake == 0:
        raise ValueError("The external test set contains only one class, so the metrics "
                         "would be meaningless. Check the files in data/polyglot/.")
    return out["text"].reset_index(drop=True), out["label"].to_numpy()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_vectorizer():
    """prepare_text runs inside the vectorizer, so every saved pipeline
    applies identical preprocessing to any new dataset."""
    return TfidfVectorizer(
        preprocessor=prepare_text,  # also lowercases
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def build_classifiers():
    """Classifiers compared on the same TF-IDF features. Hyperparameters are
    fixed in advance, never tuned on validation or PolyglotFakeFacts."""
    return {
        "svm": LinearSVC(C=1.0, random_state=RANDOM_STATE, max_iter=5000),
        # min_samples_leaf=5 stops the tree from memorising single articles
        # and gives leaves a meaningful FAKE proportion for predict_proba.
        "decision_tree": DecisionTreeClassifier(min_samples_leaf=5,
                                                random_state=RANDOM_STATE),
    }


def classifier_step(model):
    return model.steps[-1][1]


def save_model(model, name):
    path = MODEL_PATHS[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(name):
    path = MODEL_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/train.py` first.")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Evaluation and reporting
# ---------------------------------------------------------------------------
def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=LABEL_FAKE, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=LABEL_FAKE, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=LABEL_FAKE, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[LABEL_TRUE, LABEL_FAKE]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[LABEL_TRUE, LABEL_FAKE], target_names=CLASS_NAMES,
            digits=4, zero_division=0),
        "n_samples": int(len(y_true)),
    }


def plot_confusion_matrix(cm, title, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_metrics(experiment, model_name, metrics, path=METRICS_PATH):
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault(experiment, {})[model_name] = metrics
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def load_metrics(path=METRICS_PATH):
    return json.loads(path.read_text()) if path.exists() else {}


def print_header(title):
    print("\n" + "=" * 50 + f"\n{title}\n" + "=" * 50 + "\n")


def print_metrics(metrics):
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("\n" + metrics["classification_report"])


def metrics_table(columns):
    """columns: dict of {column title: metrics dict} -> Markdown table."""
    names = list(columns)
    lines = ["| Metric | " + " | ".join(names) + " |",
             "|---|" + "---:|" * len(names)]
    for label, key in [("Accuracy", "accuracy"), ("Precision", "precision"),
                       ("Recall", "recall"), ("F1-score", "f1")]:
        lines.append(f"| {label} | " + " | ".join(f"{columns[n][key]:.4f}" for n in names) + " |")
    return "\n".join(lines)
