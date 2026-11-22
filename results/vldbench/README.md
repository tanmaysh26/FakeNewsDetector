# VLDBench Experiment: TF-IDF + Linear SVM (text only)

This experiment trains and tests the project's existing TF-IDF + Linear SVM on VLDBench, then compares the result with the text-only results published in the VLDBench paper. Only the Linear SVM is used. The Decision Tree from the main experiments is not part of it.

## Dataset

**VLDBench** (Raza et al., 2025; Hugging Face `vector-institute/VLDBench`, arXiv:2502.11361v5, Information Fusion 2026). It contains 31,339 news articles from May 2023 to September 2024, each paired with an image, collected from many outlets (Financial Times, USA TODAY, CNN, The Guardian, and others).

| Item | Column used | Notes |
|---|---|---|
| Headline | `headline` | No missing values |
| Article body | `article_text` | No missing values |
| Text-only label | `text_label` | `Likely` (Likely Disinformation) or `Unlikely` (Unlikely Disinformation) |

Only these text columns are read. The `image`, `image_description` and `multimodal_label` columns are never loaded. The label meaning comes from the paper (Section 3), and the official text-only script (`code/llama3.2-nlp.py` in the GitHub repository) also scores predictions against `text_label`.

**Label mapping:** `Likely` → FAKE = 1 (the positive class), and `Unlikely` → TRUE = 0. This is the same encoding as the WELFake and PolyglotFakeFacts experiments.

**Class balance (`text_label`):** 23,180 Likely (74.0%) and 8,159 Unlikely (26.0%).

**Label notes:**
- The paper's Table 4 reports 57.1% Likely / 42.9% Unlikely, but that matches `multimodal_label` (17,898 / 13,441), not `text_label`.
- The two label columns agree on only 57% of articles.
- This experiment uses `text_label`, because it is the text-only label used by the official text-only code.

## Split

The Hugging Face dataset has **a single split named `train`, with no official test split**. The paper's instruction fine-tuning used a "chronologically stratified 70/15/15 split by outlet", but that split was not released. Its zero-shot results were computed on the whole dataset.

This experiment therefore uses a reproducible **80/20 stratified random split** (`random_state=42`), the same method as the WELFake experiment:

| Part | Articles | Likely (FAKE) | Unlikely (TRUE) |
|---|---:|---:|---:|
| Train | 25,071 | 18,544 | 6,527 |
| Test | 6,268 | 4,636 | 1,632 |

The IDs of the test articles are saved in `test_ids.csv`.

**Leakage checks:**
- There are no exact duplicate articles or headlines, so no rows were removed.
- TF-IDF is fitted on the training texts only, inside `Pipeline.fit`. The test texts only go through `transform()`.
- Hyperparameters are the project's existing ones and were not changed after seeing the test results.
- All but 2 of the 6,268 test articles (99.97%) come from outlets that also appear in training. A random split does not separate outlets the way the paper's IFT split did, so the model can benefit from outlet writing style. The "outlet majority" reference row below measures how much the outlet name alone is worth.

## Preprocessing and model

Everything is identical to the main project (`src/utils.py`):
- **Input text:** `headline + " " + article_text`
- **Cleaning:** HTML unescaping and tag removal, URL removal, lowercasing, and whitespace normalisation
- **TF-IDF:** `ngram_range=(1,2)`, `min_df=2`, `max_df=0.95`, `sublinear_tf=True`. The fitted vocabulary has 1,638,951 terms.
- **Classifier:** `LinearSVC(C=1.0, max_iter=5000, random_state=42)`, with no class weighting

## Results (measured by this experiment)

Test set: 6,268 articles. Positive class: FAKE (`Likely`).

| Metric | FAKE = positive (binary) | Weighted average | Macro average |
|---|---:|---:|---:|
| Accuracy | 0.8145 | 0.8145 | 0.8145 |
| Precision | 0.8337 | 0.8042 | 0.7770 |
| Recall | 0.9357 | 0.8145 | 0.7028 |
| F1-score | 0.8818 | 0.8003 | 0.7253 |

Per class (`classification_report.txt`):

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| TRUE (Unlikely) | 0.7202 | 0.4700 | 0.5688 | 1,632 |
| FAKE (Likely) | 0.8337 | 0.9357 | 0.8818 | 4,636 |

**Confusion matrix** (rows are actual, columns are predicted; see `confusion_matrix.png`):

| | Predicted TRUE | Predicted FAKE |
|---|---:|---:|
| Actual TRUE | 767 | 865 |
| Actual FAKE | 298 | 4,338 |

**Reference baselines on the same test set** (no learning from text):

| Baseline | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| Always predict Likely | 0.7396 | 0.6289 | 0.4252 |
| Each outlet's most common training label | 0.7423 | 0.6468 | 0.4593 |

## Comparison with published text-only results

`comparison.csv` and `comparison.md` hold the full table. **Only the first three rows were produced by this experiment.** All other numbers are copied from the paper and were not re-run. All values are percentages. For this experiment's rows, precision, recall and F1 are weighted averages, because the official evaluation script (`code/eval291124.py`) uses `average="weighted"`. The paper itself does not say which averaging it used.

| Model | Result type | Setting | Evaluation data | Acc | Prec | Rec | F1 | Source |
|---|---|---|---|---:|---:|---:|---:|---|
| **TF-IDF + Linear SVM (C=1.0)** | **this experiment** | supervised | our 20% test split | **81.45** | **80.42** | **81.45** | **80.03** | `metrics.json` |
| Always Likely | this experiment (reference) | no learning | our 20% test split | 73.96 | 54.71 | 73.96 | 62.89 | `metrics.json` |
| Outlet majority | this experiment (reference) | outlet only | our 20% test split | 74.23 | 70.31 | 74.23 | 64.68 | `metrics.json` |
| LLaMA-3.2-1B-Instruct | published | zero-shot | whole dataset | 70.29 | 69.78 | 68.92 | 69.35 | Table 9 |
| Qwen2-7B-Instruct | published | zero-shot | whole dataset | 69.92 | 68.34 | 69.10 | 68.72 | Table 9 |
| Mistral-7B-Instruct-v0.3 | published | zero-shot | whole dataset | 68.58 | 68.10 | 65.09 | 66.56 | Table 9 |
| LLaMA-3.1-8B-Instruct | published | zero-shot | whole dataset | 68.21 | 62.13 | 62.10 | 62.11 | Table 9 |
| Phi-3-mini-128k-Instruct | published | zero-shot | whole dataset | 57.15 | 55.12 | 58.21 | 56.62 | Table 9 |
| GLM-4-9B-Chat | published | zero-shot | whole dataset | 51.14 | 60.19 | 53.28 | 56.52 | Table 9 |
| Vicuna-7B-v1.5 | published | zero-shot | whole dataset | 55.21 | 56.78 | 52.48 | 54.55 | Table 9 |
| DeepSeek-V2-Lite-Chat | published | zero-shot | whole dataset | 51.96 | 52.61 | 53.53 | 51.96 | Table 9 |
| InternLM2-7B | published | zero-shot | whole dataset | 51.19 | 49.85 | 51.16 | 50.50 | Table 9 |
| Phi-3-mini-128k-Instruct (IFT) | published | instruction fine-tuned | paper's 15% test split | n/a | n/a | n/a | 67.2 | Section 5.2 / Fig. 9 |

The published zero-shot numbers are means over three runs and come from Table 9 of arXiv:2502.11361v5. Table 9's vision-language models use text plus images, so they are excluded. Figure 9 reports other instruction-fine-tuned results only as bar charts, so only the one value stated in the text is included.

## Interpretation

- **The SVM beats every published text-only number in this table**: 81.5% accuracy and 80.0% weighted F1, against 70.3% / 69.4% for the best zero-shot LLM (LLaMA-3.2-1B-Instruct) and 67.2% F1 for the fine-tuned Phi-3-mini. That is expected to some degree: the SVM is trained on 25,000 labelled VLDBench articles, while the zero-shot LLMs never see a labelled example.
- **The baselines matter more than the LLM comparison.** Predicting "Likely" for every article already gives 74.0% accuracy, and knowing only the outlet gives 74.2%. The SVM's real gain over these baselines is about 7 accuracy points and 15 to 17 weighted-F1 points. Every published zero-shot model scores below the always-Likely accuracy on `text_label`. This suggests the published numbers may not have been computed on exactly this label column or class balance (see the Table 4 note above). The paper does not give enough detail to confirm.
- **The model leans towards FAKE**, like it did on PolyglotFakeFacts. It finds 94% of Likely articles but only 47% of Unlikely ones, and 865 of the 1,632 Unlikely test articles are flagged as disinformation. Accuracy hides this, so the macro F1 (0.725) is the fairer single number.
- **The SVM learns wording, not facts.** VLDBench labels depend on whether a false claim is used to persuade. The SVM cannot check claims; it learns which words and outlets tend to appear with each label. Because almost every test article's outlet is also in training, some of the score likely comes from outlet style.

## Limitations

- **The split differs from the paper's.** The comparison is indicative, not a like-for-like benchmark. The published zero-shot results use the whole dataset, and the IFT result uses an unreleased split grouped by outlet and date. This experiment uses its own random 20% test split.
- **The paper's averaging is unclear.** It does not say whether precision, recall and F1 are weighted, macro or binary. Weighted averages are used here because the official evaluation script uses them.
- **The class balance in the paper doesn't match the released labels.** The paper's 57.1/42.9 figure matches `multimodal_label`, and the released `text_label` is 74.0/26.0. The two label columns agree on only 57% of articles.
- **The labels were created by GPT-4o with human verification**, so the model is learning to agree with that annotation process.
- **Results come from a single split and a single run.** No cross-validation or repeated seeds were used.
- **The "88 outlets" vs "58 sources" difference is unexplained.** The `outlet` column has 88 distinct values, while the paper reports 58 sources. Some outlets probably appear under more than one name.

## Reproduce

1. Download the dataset file (355 MB, including the images, which this experiment ignores) into `data/vldbench/`:
   ```bash
   curl -L -o data/vldbench/train-00000-of-00001.parquet \
     https://huggingface.co/datasets/vector-institute/VLDBench/resolve/main/data/train-00000-of-00001.parquet
   ```
2. Install the requirements. `pyarrow` is needed to read the Parquet file.
   ```bash
   pip install -r requirements.txt
   ```
3. Run the experiment (about 1 minute):
   ```bash
   python src/vldbench_experiment.py
   ```

## Files

| File | Contents |
|---|---|
| `metrics.json` | Split details, binary, weighted and macro metrics, confusion matrix, and baselines |
| `classification_report.txt` | Per-class precision, recall and F1 |
| `confusion_matrix.png` | Confusion matrix plot |
| `comparison.csv`, `comparison.md` | This experiment's results next to the published text-only results |
| `test_ids.csv` | `unique_id` of every test article |
| `../../models/vldbench/fake_news_tfidf_svm_vldbench.joblib` | The trained pipeline (59 MB). Not in git; rebuilt by the script. |
