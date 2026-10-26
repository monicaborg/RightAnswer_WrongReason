# Right Answer, Wrong Reason

**Explainable AI for Automated Conceptual Understanding Assessment**

Codebase for an EGH490 capstone research project completed by Monica Borg
(n9802045) under the supervision of Dr Sam Cunningham and Dr Wageeh Boles,
QUT Faculty of Engineering, 2026. The project applies explainable AI to the
automated scoring of students' conceptual reasoning in Signals & Systems.

> **The student response data is not included in this repository.** It is
> governed by QUT ethics approval and cannot be shared publicly. See
> [Data access and ethics](#data-access-and-ethics).

---

## Project overview

Concept inventories measure whether a student picked the right answer, not
whether they reasoned their way to it. QUT's Check Conceptual Understanding
(CCU) quizzes for Signals & Systems ask students to justify each
multiple-choice answer in free text, and earlier work showed those
justifications can be classified automatically with high accuracy (Somers,
Cunningham & Boles, 2021). Those classifiers are black boxes, though: they
return a valid/invalid label with no account of why, and students piloting the
system in 2018 asked for exactly that explanation.

This project:

1. **Re-annotated** all 3,386 free-text responses across six CCU questions
   (2015–2020) for reasoning **validity** and expressed **confidence**, after
   the original labelled dataset could not be located.
2. **Rebuilt the transformer ensemble** (RoBERTa, ALBERT, XLNet, soft voting)
   and benchmarked it against the published baseline.
3. **Added an explainable-AI layer**: LIME and SHAP attributions at word
   (unigram) and phrase (bigram) level, with attention weights as a comparison.
4. **Measured whether the explanations can be trusted**, using faithfulness
   (ERASER comprehensiveness and sufficiency), stability and coverage metrics.
5. **Analysed where the classifier fails**, using a four-category
   cross-tabulation of reasoning validity against MCQ correctness (genuine
   understanding, slip, guess, consistent misconception), building on the
   Concept Understanding Matrix (Boles, Goncher & Jayalath, 2015) and the
   conceptual-understanding pointers of Cunningham et al. (2017).

---

## What the project found

### Annotation

- The three largest CCUs (CCU1–3, 300 sampled responses) were independently
  reviewed by Dr Sam Cunningham: **86.7% raw agreement, Cohen's κ = 0.722**
  (substantial).
- 80% of disagreements traced to one identifiable convention: a formula or
  concept stated without being applied (e.g. "1/T = f" on its own) is marked
  invalid. A matched case study on CCU6 found this stricter convention carries
  **no measurable accuracy cost** compared with the original, more lenient
  labels.

### Classification (validity, three-model ensemble, 5-fold CV)

| CCU | Accuracy | AUC | Somers et al. accuracy | Somers et al. AUC |
|-----|---------:|----:|-----------------------:|------------------:|
| CCU1 | 95.9% | 0.966 | 97.97% | 0.8641 |
| CCU2 | 97.0% | 0.989 | 97.34% | 0.9747 |
| CCU3 | 89.2% | 0.950 | 98.66% | 0.9814 |
| CCU4 | 96.4% | 0.983 | 96.86% | 0.9661 |
| CCU5 | 92.5% | 0.960 | 93.54% | 0.9376 |
| CCU6 | 89.8% | 0.964 | 95.95% | 0.9577 |
| **Mean** | **93.47%** | **0.9687** | **96.72%** | **0.9469** |

- The ensemble trails the published baseline on accuracy but **exceeds it on
  AUC in five of six CCUs**. AUC measures whether the model separates valid
  from invalid reasoning regardless of decision threshold, so most of the
  accuracy gap is a threshold-placement effect.
- **Confidence** classification reached **97.94% accuracy and 0.9931 AUC**,
  and was far more consistent across CCUs (1.6-point accuracy spread vs 7.8 for
  validity).
- **ELECTRA-small** failed to learn a usable decision boundary under every
  configuration tested and was excluded from the ensemble.
- **Ensembling was not uniformly beneficial**: it beat its best individual
  member on only two of six CCUs, with a weak XLNet pulling the vote down on
  CCU3 and CCU6.
- **Keeping duplicate responses helped**: deduplication removed ~19% of the
  data and lowered accuracy in every comparison tested (e.g. ALBERT on CCU2:
  90.7% deduplicated vs 93.6% full).

### Explanation quality (150 sampled responses per CCU, means across six CCUs)

| Metric | LIME validity (uni / bi) | SHAP validity (uni / bi) | LIME confidence (uni / bi) | SHAP confidence (uni / bi) |
|--------|-------------------------:|-------------------------:|---------------------------:|---------------------------:|
| Comprehensiveness | 0.435 / 0.400 | 0.607 / 0.700 | 0.372 / 0.353 | 0.529 / 0.639 |
| Sufficiency | 0.750 / 0.656 | 0.926 / 0.960 | 0.804 / 0.707 | 0.969 / 0.979 |
| Stability | 0.905 / 0.809 | 1.000 / 1.000 | 0.913 / 0.761 | 1.000 / 1.000 |
| Coverage | 0.531 / 0.520 | 0.590 / 0.559 | 0.634 / 0.632 | 0.756 / 0.658 |

- **SHAP is more faithful than LIME** in every condition and perfectly stable.
  Both clear the 0.2–0.5 comprehensiveness range typical of faithful
  attribution on real text.
- **LIME and SHAP agree strongly** on which words matter (Pearson r = 0.72–0.83
  per CCU, mean 0.770, across 5,651 shared tokens), despite resting on
  entirely different foundations.
- **Phrase-level attribution helps SHAP and hurts LIME**, and its effect on
  coverage depends on the question, so granularity is best chosen per question.
- **Attention is not a reliable explanation**: ALBERT's highest attention
  falls on the final token or punctuation in 88.6% of responses regardless of
  content, reproducing Jain & Wallace (2019) on real student writing. The
  severity is architecture-dependent (RoBERTa 43.8%, XLNet 28.8%).
- **Explanation quality tracks the task, not just accuracy**: confidence
  explanations have higher coverage and sufficiency but lower
  comprehensiveness than validity, at both granularities.
- **The classifier attends to concept vocabulary**: on CCU1 and CCU3, 72–75% of
  explained responses have a domain concept word among their top three
  attributions, with no evidence of reliance on surface features such as
  length or fluency.

### MCQ–free-text analysis

| | MCQ correct | MCQ incorrect |
|---|---|---|
| **Reasoning valid** | Genuine understanding (40.6%) | Slip (1.2%) |
| **Reasoning invalid** | Guess (30.5%) | Consistent misconception (27.6%) |

- About 43% of correct MCQ answers are paired with invalid reasoning, and
  guesses outnumber genuine understanding on CCU2 and CCU4, so MCQ correctness
  alone overstates understanding.
- A two-CCU pilot that fed the MCQ selection to the model alongside the text
  was inconclusive (no effect on CCU3, degradation on CCU6).

The full results, methodology and limitations are in the final report.

---

## Repository layout

The repository contains code only. The corpus, trained checkpoints and all
generated outputs are excluded (see
[Data access and ethics](#data-access-and-ethics)).

```
RightAnswer_WrongReason/
├── configs/                YAML configs (data, model, training, xai)
├── egh490/                 Python package
│   ├── data/               Schema, CSV loading, label encoding, k-fold splits,
│   │                       optional MCQ-augmented input, dataset preparation
│   ├── models/             TransformerClassifier, Trainer, Ensemble
│   ├── xai/                LIME, SHAP, attention, quality metrics, plots
│   ├── evaluation/         Inter-rater agreement (Cohen's kappa)
│   └── utils/              Seeding, logging, I/O, config, device detection
├── scripts/
│   ├── training/           train_all_models.py, benchmark_batch_sizes.py
│   ├── ensemble/           evaluate_ensemble.py (inference-only scoring)
│   ├── explanation/        explain.py, mcq_mismatch_analysis.py
│   ├── plotting/           plot_results.py, plot_xai.py
│   ├── evaluation/         compute_agreement.py
│   ├── reporting/          export_educator_report.py
│   └── archive/            Superseded scripts, kept for provenance
├── tests/                  62 tests (unit + integration)
├── data/                   Gitignored: corpus not distributed
└── outputs/                Gitignored: regenerated by the pipeline
```

The package can be used directly:

```python
from egh490.models import TransformerClassifier, Trainer, Ensemble
from egh490.data import DataModule
from egh490.xai import LimeExplainer, ShapExplainer, AttentionExtractor
from egh490.utils import set_global_seed, load_config, get_logger
```

---

## Replicating the project

Replication needs the annotated corpus, which must be obtained separately
(see [Data access and ethics](#data-access-and-ethics)). Place it at
`data/raw/signals_systems_validity_corpus.csv`.

### 1. Set up

```bash
git clone https://github.com/monicaborg/RightAnswer_WrongReason.git
cd RightAnswer_WrongReason
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -v        # 62 tests
```

Python 3.10 or 3.11 is required. Dependency versions are pinned in
`pyproject.toml`.

### 2. Train (per CCU, per task)

```bash
# Validity
python scripts/training/train_all_models.py \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --task validity --models roberta albert xlnet --save-models

# Confidence
python scripts/training/train_all_models.py \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --task confidence --models roberta albert xlnet --save-models
```

Repeat for `ccu1` to `ccu6`. Add `electra` to `--models` to reproduce the
ELECTRA result.

### 3. Score the ensemble (inference only)

```bash
python scripts/ensemble/evaluate_ensemble.py \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --task validity \
  --checkpoint-dir outputs/checkpoints \
  --dataset-tag signals_systems_validity_corpus_ccu1
```

This rebuilds the training folds and soft-votes the saved checkpoints on each
fold's held-out set, without retraining.

### 4. Generate and evaluate explanations

```bash
# Word level
python scripts/explanation/explain.py --ensemble \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --task validity --n-samples 150 \
  --checkpoint-dir outputs/checkpoints \
  --dataset-tag signals_systems_validity_corpus_ccu1

# Phrase level
python scripts/explanation/explain.py --ensemble \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --ccu ccu1 --task validity --n-samples 150 --ngram 2 \
  --checkpoint-dir outputs/checkpoints \
  --dataset-tag signals_systems_validity_corpus_ccu1
```

ELECTRA is excluded from the ensemble by default. Outputs are written to
`outputs/explanations/<unigram|bigram>/<task>/<ccu>/`, so runs never overwrite
each other.

### 5. Analysis, figures and educator export

```bash
# Validity x MCQ cross-tabulation and classifier error breakdown
python scripts/explanation/mcq_mismatch_analysis.py \
  --csv data/raw/signals_systems_validity_corpus.csv --ccu ccu1 \
  --checkpoint-dir outputs/checkpoints \
  --dataset-tag signals_systems_validity_corpus_ccu1

# Figures from saved explanations (no recomputation)
python scripts/plotting/plot_xai.py \
  --explanations-dir outputs/explanations/unigram/validity/ccu1 \
  --out-dir outputs/figures/xai_ccu1

# Per-response CSV for educators
python scripts/reporting/export_educator_report.py \
  --csv data/raw/signals_systems_validity_corpus.csv \
  --explanations-dir outputs/explanations/unigram/validity/ccu1 \
  --ccu ccu1 --output outputs/educator_reports/ccu1_report.csv
```

### Key options

```
train_all_models.py
  --ccu NAME                  ccu1–ccu6
  --task {validity,confidence}
  --models [NAMES]            any of roberta albert xlnet electra
  --epochs N                  max epochs (default 6)
  --lr FLOAT                  learning rate (default 2e-5)
  --patience N                early-stopping patience (default 2; 0 disables)
  --class-weighted            inverse-frequency class weights (not used in final runs)
  --include-mcq               prepend the MCQ selection to the text (pilot only)
  --save-models               keep checkpoints for ensemble scoring and XAI

explain.py
  --ensemble                  explain the ensemble rather than one model
  --exclude-models [NAMES]    default: electra
  --n-samples N               responses to explain (150 used in the project)
  --ngram {1,2}               word- or phrase-level attribution
  --no-lime / --no-shap / --no-attention / --no-eval
```

### Configuration used in the project

| Setting | Value |
|---------|-------|
| Corpus | Full 3,386 responses (no deduplication, single-word responses retained) |
| Learning rate | 2×10⁻⁵ |
| Epochs | max 6, early stopping (patience 2 on macro-F1) |
| Batch size | RoBERTa 8, XLNet 4, ALBERT 16, ELECTRA 16 |
| Cross-validation | 5-fold stratified, reported as fold mean |
| Seed | 20260413 (`egh490/utils/seeding.py`) |
| Labels | validity: incorrect=0, correct=1; confidence: low=0, high=1 |

### Hardware and run time

Developed on a MacBook Pro (Apple Silicon, 16 GB RAM, MPS backend). Training
the full grid (both tasks, six CCUs, five folds) takes roughly 70–80 hours on
this hardware, with XLNet at batch size 4 the largest share. Explanation
generation takes about 2–2.5 hours per CCU, per task, per granularity.

### What to expect from a re-run

Fixed seeds make data splits and response sampling deterministic. The same 150
responses are explained for every task and granularity. Model weights, however,
are not bit-identical across runs, because floating-point operations on
GPU/MPS hardware are not deterministic.

An independent retrain of all three models across all six CCUs and both tasks
(36 conditions) reproduced accuracy to within a mean of **1.07 percentage
points**. RoBERTa (0.23) and ALBERT (0.61) reproduced tightly; XLNet varied
more (2.38, maximum 7.0). Expect re-runs to match the trends and approximate
magnitudes above, not every decimal place.

---

## Data schema

The pipeline expects a CSV with these columns (defined in
`egh490/data/schema.py`):

| Column | Description | Used by |
|--------|-------------|---------|
| `uid` | Unique response ID | Metadata |
| `ccuname` | CCU identifier (ccu1–ccu6) | Filtering, analysis |
| `time` | Submission timestamp | Metadata |
| `q1mcr` | MCQ answer selected (a–d) | Analysis, `--include-mcq` |
| `q1txr` | **Free-text response** | **Model input** |
| `q2mcr`, `q3mcr` | Other MCQ answers | Metadata |
| `validity` | correct / incorrect | **Label (validity task)** |
| `confidence` | high / low | **Label (confidence task)** |

---

## Data access and ethics

This project was conducted under QUT Human Research Ethics approval
**#11077** (LR 2026-11077-29139, approved 24 April 2026) as a secondary
analysis of existing, de-identified student responses.

Because the corpus contains student-written text, **no data is included in
this repository**: the annotated corpus, trained checkpoints, generated
explanations, educator reports and agreement analyses are all excluded
(`data/` and `outputs/` are gitignored in full).

**Requests for access to the data should be directed to Dr Sam Cunningham at
QUT**, the chief investigator on the ethics approval. Any access is subject to
QUT's ethics and data-management requirements.

---

## Key references

- Somers, R., Cunningham, S., & Boles, W. (2021). Applying natural language
  processing to automatically assess student conceptual understanding from
  textual responses. *Australasian Journal of Educational Technology, 37*(5),
  98–115.
- Boles, W., Goncher, A., & Jayalath, D. (2015). Uncovering misconceptions
  through text analysis. *Research in Engineering Education Symposium*.
- Cunningham, S., Goncher, A., Mukherjee, M., & Boles, W. (2017). Pointers to
  conceptual understanding. *AAEE Annual Conference*.
- Cunningham, S., Mukherjee, M., Goncher, A., Boles, W., Swan, J., & Harris, T.
  (2018). Regular and automatic feedback of concepts as formative assessment.
  *AAEE Annual Conference*.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?":
  Explaining the predictions of any classifier. *KDD*.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting
  model predictions. *NeurIPS*.
- DeYoung, J., et al. (2019). ERASER: A benchmark to evaluate rationalized NLP
  models. *ACL*.
- Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *NAACL*.
- Wiegreffe, S., & Pinter, Y. (2019). Attention is not not explanation.
  *EMNLP-IJCNLP*.
- Gunasekara, S., & Saarela, M. (2025). Explainable AI in education:
  Techniques and qualitative assessment. *Applied Sciences, 15*(3), 1239.

The full reference list is in the final report.
