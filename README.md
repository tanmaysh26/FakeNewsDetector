# Fake News Detection with TF-IDF: Comparing Four Classifiers

## 1. Objective

This project runs two experiments that compare four classifiers trained on the same TF-IDF features: a Linear SVM, a Decision Tree, Logistic Regression and Multinomial Naive Bayes.

- **Experiment 1:** How well does each model perform on a held-out 20% of WELFake?
- **Experiment 2:** How well does each WELFake-trained model generalize to the unseen PolyglotFakeFacts dataset (cross-dataset external evaluation)?

All four classifiers learn statistical word and phrase patterns from labeled news data. It **does not verify the factual truth of any claim**. A "FAKE" prediction means the text resembles articles labeled fake in WELFake, and says nothing about whether the claims in it are false.

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
        → TF-IDF (fit once, on train only) ─┬→ LinearSVC (C=1.0)                → validation metrics
                                            ├→ DecisionTreeClassifier          → validation metrics
                                            │  (min_samples_leaf=5)
                                            ├→ LogisticRegression (C=1.0)      → validation metrics
                                            └→ MultinomialNB (alpha=1.0)       → validation metrics
```

One TF-IDF vectorizer is fitted on the training split, and all four classifiers are trained on its features. The only difference between the models is the classifier, so the comparison is like-for-like.

Exact duplicate articles are removed before splitting. Otherwise, identical copies could appear in both the training and validation sets and inflate the scores.

## 4. Experiment 2 methodology

```
Saved pipelines (TF-IDF + each of the four classifiers; trained on the WELFake training split)
        → PolyglotFakeFacts (headline + English text) → predict → external metrics
```

The pipelines trained and validated in Experiment 1 are saved to `models/fake_news_tfidf_<model>.joblib` (`svm`, `decision_tree`, `logistic_regression`, `naive_bayes`) and reused unchanged in Experiment 2. PolyglotFakeFacts is only passed through `predict()`, which applies the stored preprocessing, the stored vocabulary, and the stored classifier.

## 5. TF-IDF

TF-IDF represents each document as a sparse vector of word and bigram weights. Terms that are frequent in a document but rare across the corpus get higher weights. The settings are:

- `ngram_range=(1,2)`: unigrams and bigrams.
- `min_df=2`: ignore terms that appear in fewer than 2 training documents.
- `max_df=0.95`: ignore terms that appear in more than 95% of documents.
- `sublinear_tf=True`: use `1 + log(tf)` so that long, repetitive articles do not dominate.

Preprocessing is deliberately light: HTML unescaping and tag removal, URL removal, lowercasing, and whitespace normalization. There is no stemming or stopword removal, because bigrams and stylistic function words can carry a useful signal. The preprocessing function runs inside the vectorizer, so both datasets receive exactly the same preprocessing.

## 6. Classifiers

### Linear SVM

TF-IDF produces very high-dimensional, sparse features, often hundreds of thousands to millions of dimensions. In a space that large, text classes are usually close to linearly separable. That makes a linear max-margin classifier accurate, fast, and memory-efficient. Kernel SVMs such as RBF scale poorly with the number of samples and add little for this kind of data.

### Decision Tree

The Decision Tree is the comparison model. It classifies an article by asking a sequence of yes/no questions about single features, such as "is the TF-IDF weight of `reuters` above 0.01?", until it reaches a leaf. That makes it easy to interpret, but it is a poor fit for sparse text: each split looks at one term out of 1.6 million, most terms are zero in most articles, and the tree tends to latch onto a few very strong source markers. It is therefore expected to overfit WELFake's artifacts more than the SVM, which weighs every term at once.

The only setting is `min_samples_leaf=5`, chosen in advance. It stops the tree from creating a leaf for every single training article, and it means each leaf's share of FAKE articles is a usable score. No depth limit is set, and nothing is tuned on validation or PolyglotFakeFacts.

### Logistic Regression

Logistic Regression is also a linear model over all TF-IDF terms, like the SVM, but it is trained to output a probability of FAKE (log loss) instead of a maximum-margin score (hinge loss). Its weights can be read the same way as the SVM's. Settings: `C=1.0` (the default, matching the SVM), `solver="liblinear"`, `max_iter=1000`.

### Multinomial Naive Bayes

Naive Bayes is the classic text-classification baseline. Instead of learning a boundary, it estimates how strongly each term is associated with each class from its total weight in that class, and assumes the terms are independent. It trains in seconds. Settings: `alpha=1.0` (standard Laplace smoothing). It needs non-negative features, which TF-IDF provides.

All hyperparameters above are library defaults or fixed in advance, and none were tuned on validation or PolyglotFakeFacts.

## 7. Installation

The datasets and trained model are not included in this repository, because of file size limits and dataset licences. Download the data as described in section 2, then run the scripts to rebuild the models.


```bash
pip install -r requirements.txt
```

## 8. Run Experiment 1

```bash
python src/train.py
```

This trains all four models (about 5 minutes) and produces one `models/fake_news_tfidf_<model>.joblib` file per model, one confusion matrix per model (`results/welfake_validation_confusion_matrix_<model>.png`), the Experiment 1 entries in `results/metrics.json`, and a side-by-side comparison table.

## 9. Run Experiment 2

```bash
python src/evaluate_external.py
```

This evaluates all four saved models and produces one confusion matrix per model (`results/polyglot_external_confusion_matrix_<model>.png`), the Experiment 2 entries in `results/metrics.json`, and a table comparing all four models on both datasets. Run Experiment 1 first.

### Classifying a single article (optional utility)

After running Experiment 1, you can classify any article from the command line with all four models. This uses the saved models as-is and is not part of the two experiments.

```bash
python src/predict.py --text "Article text..."
python src/predict.py --title "Headline" --text "Article text..."
python src/predict.py --file article.txt
python src/predict.py        # paste the article, then Ctrl-D (Windows: Ctrl-Z, Enter)
```

The output shows one line per model with its predicted label (FAKE or TRUE) and its score:

- **Linear SVM:** the decision score. Positive means FAKE, negative means TRUE, and values further from 0 indicate more confidence. It is not a probability.
- **Decision Tree:** P(FAKE), the share of FAKE training articles in the leaf the article lands in, from 0 to 1. Above 0.5 means FAKE.
- **Logistic Regression and Naive Bayes:** P(FAKE), the model's probability that the article is FAKE, from 0 to 1. Above 0.5 means FAKE.

The output also reports how many terms matched the training vocabulary and warns when very few do. Predictions reflect similarity to WELFake's writing patterns, not fact-checking.

## 10. Evaluation metrics

The scripts report accuracy, precision, recall, and F1 with FAKE as the positive class, plus a full classification report (per-class and macro/weighted averages) and a confusion matrix. All numbers are computed from the actual data and saved in `results/metrics.json`:

| Model | Dataset | Accuracy | Precision | Recall | F1-score | TRUE recall |
|---|---|---:|---:|---:|---:|---:|
| Linear SVM | WELFake | **0.9724** | **0.9679** | **0.9713** | **0.9696** | 0.9733 |
| Decision Tree | WELFake | 0.9345 | 0.9293 | 0.9261 | 0.9277 | 0.9415 |
| Logistic Regression | WELFake | 0.9556 | 0.9496 | 0.9528 | 0.9512 | 0.9580 |
| Naive Bayes | WELFake | 0.8513 | 0.9481 | 0.7111 | 0.8127 | 0.9677 |
| Linear SVM | Polyglot | 0.6417 | 0.5828 | **0.9039** | 0.7086 | 0.3977 |
| Decision Tree | Polyglot | 0.5156 | 0.4986 | 0.8426 | 0.6265 | 0.2113 |
| Logistic Regression | Polyglot | **0.6771** | 0.6209 | 0.8479 | **0.7169** | 0.5182 |
| Naive Bayes | Polyglot | 0.5395 | **0.6763** | 0.0859 | 0.1525 | 0.9617 |

Precision, recall and F1 use FAKE as the positive class; TRUE recall is the share of real articles correctly labelled TRUE.

WELFake validation uses 12,736 articles (50,940 for training). The PolyglotFakeFacts test uses 10,190 of its 10,206 articles. 76 have no English translation; 60 of these are scored on their headline alone, and 16 (all real) have no text at all and are dropped.

**On WELFake the Linear SVM is best**, followed by Logistic Regression (95.6%), the Decision Tree (93.5%) and Naive Bayes (85.1%).

**On PolyglotFakeFacts, Logistic Regression generalises best** (67.7% accuracy, F1 0.717), ahead of the SVM (64.2%, F1 0.709). The ranking of the two linear models flips between datasets: the SVM fits WELFake more tightly, and that tighter fit transfers slightly worse. The Decision Tree is near chance (51.6%). It relies on a handful of individual terms, while the linear models spread their decision over every term in the article.

The SVM, tree and Logistic Regression all lean towards FAKE on PolyglotFakeFacts. The SVM catches 90% of fake articles but flags 60% of real ones as fake (3,179 of 5,278). Logistic Regression flags 48% of real articles (2,543), and the tree flags 79%. Section 12 explains why.

**Naive Bayes fails in the opposite direction.** Already on WELFake it misses 29% of fake articles (recall 0.71) while rarely flagging real ones. On PolyglotFakeFacts it labels 94% of all articles TRUE and catches only 422 of 4,912 fake articles (recall 0.09). Its 54% accuracy is therefore close to always predicting TRUE (51.8%). A likely cause is that with 1.6 million features and small TF-IDF weights, `alpha=1.0` smoothing swamps much of the per-term evidence, which leaves the class prior (TRUE is the larger class) with more influence. The smoothing was not tuned, by design, so this is the result of the default setting rather than the best Naive Bayes can do.

## 11. Data leakage prevention

- TF-IDF is fitted only on the WELFake training split, once, in `train_models()`, and all four classifiers share it. If the vocabulary and IDF weights were fitted on validation or test texts, information about those texts would leak into the features, and the scores would overstate performance on truly new data.
- The validation set and PolyglotFakeFacts only ever go through `transform()`, via `predict()`.
- PolyglotFakeFacts is never used for training, vocabulary construction, hyperparameter tuning, model selection, or threshold selection. As an extra check, `evaluate_external.py` asserts that the vocabulary size is unchanged after prediction.
- Exact duplicates are removed from WELFake before splitting.
- Hyperparameters are fixed in advance and are not modified based on the PolyglotFakeFacts results.

## 12. Limitations

- The model learns surface lexical and stylistic patterns, including source-specific artifacts from the four datasets merged into WELFake. It does not check facts.
- The SVM weights show that these source artifacts dominate. The strongest TRUE features are `reuters` (−13.0), `breitbart`, `said`, `washington reuters`, and `new york times`. The strongest FAKE features are `via` (+10.9), `read more`, `video`, `featured image`, `getty images`, `2016`, and `hillary`. In other words, the model largely recognises the publisher and the 2016-era topics rather than anything inherent to fake news. Real PolyglotFakeFacts articles lack the Reuters dateline and WELFake's house style, so they lose the strongest TRUE signal, which explains most of the false positives in Experiment 2.
- The Decision Tree shows the same dependence even more directly. Its first split is on `reuters`, followed by `york times` and `breitbart`, and any article containing `featured image` goes straight to a pure FAKE leaf. The fully grown tree is 141 levels deep with 804 leaves. Because it decides on a few terms at a time, an article that lacks those markers falls through to deep, narrow branches that do not transfer to new sources.
- Only exact duplicates are removed from WELFake. Near-duplicates, such as the same wire story with minor edits, can still fall on both sides of the split, so the validation scores may be slightly optimistic.
- URL removal only catches links that start with `http(s)://` or `www.`. Links such as `pic.twitter.com/...` stay in the text, which is why `twitter com` appears as a FAKE feature.
- WELFake is mostly US news from around 2016–2018. PolyglotFakeFacts covers newer European and geopolitical topics, so topic and time shift are expected to lower the external scores.
- PolyglotFakeFacts is evaluated through machine-translated English text, which may introduce translation artifacts.
- The label definitions differ between the datasets. PolyglotFakeFacts fake articles are defined by source (EUvsDisinfo-flagged outlets), which is not the same as WELFake's labeling.
- Bigram TF-IDF on the full WELFake dataset needs several GB of RAM during fitting.

## 13. Additional experiment: VLDBench (Linear SVM only)

`src/vldbench_experiment.py` trains and tests the same TF-IDF + Linear SVM on VLDBench (Raza et al., 2025), using text only: headline + article text, with the `text_label` column as the label. VLDBench has no official test split, so it uses an 80/20 stratified split (`random_state=42`). On the 6,268 test articles it scores 81.45% accuracy and 80.03% weighted F1. The best published zero-shot text-only LLM (LLaMA-3.2-1B-Instruct) reports 70.29% and 69.35%. Two caveats: always predicting the majority class already reaches 73.96% accuracy, and the published numbers use a different evaluation set.

The dataset, split, full results, comparison table, limitations and reproduction steps are in [`results/vldbench/README.md`](results/vldbench/README.md). This experiment writes only to `results/vldbench/` and `models/vldbench/`, so the WELFake and PolyglotFakeFacts results above are unchanged.
