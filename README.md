# Fake News Detection with TF-IDF + Linear SVM

## 1. Objective

This project runs two experiments with a single TF-IDF + Linear SVM classifier:

- **Experiment 1:** How well does the model perform on a held-out 20% of WELFake?
- **Experiment 2:** How well does the WELFake-trained model generalize to the unseen PolyglotFakeFacts dataset (cross-dataset external evaluation)?

The classifier learns statistical word and phrase patterns from labeled news data. It **does not verify the factual truth of any claim**. A "FAKE" prediction means the text resembles articles labeled fake in WELFake, and says nothing about whether the claims in it are false.

## 2. Datasets

**WELFake** (Verma et al., 2021; Zenodo doi:10.5281/zenodo.4561253). This is an English dataset of about 72k articles, merged from Kaggle, McIntire, Reuters, and BuzzFeed Political. It has four columns: serial number, `title`, `text`, and `label`. The documentation says 0 = fake and 1 = real, but the CSV itself is the reverse: 21,270 articles with the Reuters dateline `(Reuters)` (real news) have label 0 versus 19 with label 1, and the class sizes match the documented 35,028 real / 37,106 fake only with 0 = real. The code therefore uses **0 = real, 1 = fake**. This check used WELFake alone; it was prompted by the first external run scoring below chance, which is disclosed here for transparency. Download `WELFake_Dataset.csv` from Zenodo into `data/welfake/`.

**PolyglotFakeFacts v2.0** (Ciobanu, 2026; Mendeley Data, doi:10.17632/8yfrm6z9dx.1). This multilingual dataset contains 10,206 articles (4,912 fake, 5,294 real) in 18 languages, each with an English translation. Fake articles come from outlets flagged by EUvsDisinfo, and real articles come from credible sources. Download only `Fake.xlsx` and `Real.xlsx` into `data/polyglot/`. Do not add `20.xlsx` and `80.xlsx`, which appear to be a split of the same articles and would duplicate them. The two files use different header capitalisation, so the loader normalises column names. The fake file's label column is used where present, and otherwise each row's class comes from its file.

The exact PolyglotFakeFacts column headers and label strings are not given in the documentation. For that reason, `load_polyglot()` detects the columns, prints what it chose, and prints the raw label values, missing values, and language counts. It stops with an error if the labels are not recognizable fake/real values. To override any detected column, edit `POLYGLOT_COLUMNS` in `src/utils.py`. For non-string labels, edit `POLYGLOT_LABEL_MAP`, but only after checking the documentation.

Experiment 2 uses the **English translated text**, because the model's vocabulary comes from English-only WELFake.

Both datasets are mapped to a common encoding: `TRUE = 0` and `FAKE = 1`. FAKE is the positive class.

## 3. Experiment 1 methodology

```
WELFake → combine title + text → drop empty and exact-duplicate rows
        → 80/20 stratified split (random_state=42)
        → TF-IDF (fit on train only) → LinearSVC (C=1.0) → validation metrics
```

Exact duplicate articles are removed before splitting. Otherwise, identical copies could appear in both the training and validation sets and inflate the scores.

## 4. Experiment 2 methodology

```
Saved pipeline (TF-IDF + SVM, trained on the WELFake training split)
        → PolyglotFakeFacts (headline + English text) → predict → external metrics
```

The pipeline trained and validated in Experiment 1 is the one saved to `models/fake_news_tfidf_svm.joblib` and reused unchanged in Experiment 2. PolyglotFakeFacts is only passed through `predict()`, which applies the stored preprocessing, the stored vocabulary, and the stored SVM weights.

## 5. TF-IDF

TF-IDF represents each document as a sparse vector of word and bigram weights. Terms that are frequent in a document but rare across the corpus get higher weights. The settings are:

- `ngram_range=(1,2)`: unigrams and bigrams.
- `min_df=2`: ignore terms that appear in fewer than 2 training documents.
- `max_df=0.95`: ignore terms that appear in more than 95% of documents.
- `sublinear_tf=True`: use `1 + log(tf)` so that long, repetitive articles do not dominate.

Preprocessing is deliberately light: HTML unescaping and tag removal, URL removal, lowercasing, and whitespace normalization. There is no stemming or stopword removal, because bigrams and stylistic function words can carry a useful signal. The preprocessing function runs inside the vectorizer, so both datasets receive exactly the same preprocessing.

## 6. Linear SVM

TF-IDF produces very high-dimensional, sparse features, often hundreds of thousands to millions of dimensions. In a space that large, text classes are usually close to linearly separable. That makes a linear max-margin classifier accurate, fast, and memory-efficient. Kernel SVMs such as RBF scale poorly with the number of samples and add little for this kind of data.

## 7. Installation

The datasets and trained model are not included in this repository, because of file size limits and dataset licences. Download the data as described in section 2, then run the scripts to rebuild the model.


```bash
pip install -r requirements.txt
```

## 8. Run Experiment 1

```bash
python src/train.py
```

This produces `models/fake_news_tfidf_svm.joblib`, `results/welfake_validation_confusion_matrix.png`, and the Experiment 1 entry in `results/metrics.json`.

## 9. Run Experiment 2

```bash
python src/evaluate_external.py
```

This produces `results/polyglot_external_confusion_matrix.png`, the Experiment 2 entry in `results/metrics.json`, and a comparison table. Run Experiment 1 first.

## Classifying a single article (optional utility)

After running Experiment 1, you can classify any article from the command line. This uses the saved model as-is and is not part of the two experiments.

```bash
python src/predict.py --text "Article text..."
python src/predict.py --title "Headline" --text "Article text..."
python src/predict.py --file article.txt
python src/predict.py        # paste the article, then Ctrl-D (Windows: Ctrl-Z, Enter)
```

The output shows the predicted label (FAKE or TRUE) and the SVM decision score, where positive means FAKE, negative means TRUE, and values further from 0 indicate more confidence. The score is not a probability. The output also reports how many terms matched the training vocabulary and warns when very few do. Predictions reflect similarity to WELFake's writing patterns, not fact-checking.

## 10. Evaluation metrics

The scripts report accuracy, precision, recall, and F1 with FAKE as the positive class, plus a full classification report (per-class and macro/weighted averages) and a confusion matrix. All numbers are computed from the actual data. Fill in the table below from the console output or `results/metrics.json`:

| Metric | WELFake Validation | PolyglotFakeFacts External Test |
|---|---:|---:|
| Accuracy | | |
| Precision | | |
| Recall | | |
| F1-score | | |

## 11. Data leakage prevention

- TF-IDF is fitted only on the WELFake training split, inside `Pipeline.fit`. If the vocabulary and IDF weights were fitted on validation or test texts, information about those texts would leak into the features, and the scores would overstate performance on truly new data.
- The validation set and PolyglotFakeFacts only ever go through `transform()`, via `predict()`.
- PolyglotFakeFacts is never used for training, vocabulary construction, hyperparameter tuning, model selection, or threshold selection. As an extra check, `evaluate_external.py` asserts that the vocabulary size is unchanged after prediction.
- Exact duplicates are removed from WELFake before splitting.
- Hyperparameters are fixed in advance and are not modified based on the PolyglotFakeFacts results.

## 12. Limitations

- The model learns surface lexical and stylistic patterns, including source-specific artifacts from the four datasets merged into WELFake. It does not check facts.
- WELFake is mostly US news from around 2016–2018. PolyglotFakeFacts covers newer European and geopolitical topics, so topic and time shift are expected to lower the external scores.
- PolyglotFakeFacts is evaluated through machine-translated English text, which may introduce translation artifacts.
- The label definitions differ between the datasets. PolyglotFakeFacts fake articles are defined by source (EUvsDisinfo-flagged outlets), which is not the same as WELFake's labeling.
- Bigram TF-IDF on the full WELFake dataset needs several GB of RAM during fitting.
