# FraudShield AI

FraudShield AI is a fraud detection portfolio project demonstrating ML development,
deep learning experiments, feature engineering, inference, and AI/ML quality
engineering. `FraudShield_Final_Completed.ipynb` remains the source of truth and is
preserved unchanged at the repository root.

## Business problem

Identify fraudulent credit-card transactions while controlling missed fraud and
unnecessary alerts. Severe class imbalance makes accuracy alone misleading;
precision, recall, F1, ROC-AUC, and average precision guide evaluation.
The model produces a **Fraud Risk Score**, not a demonstrated perfectly calibrated
real-world probability. A low score does not guarantee legitimacy.

## Dataset and feature engineering

The notebook uses the Kaggle Credit Card Transactions Fraud Detection dataset:

| Split | Transactions | Period |
| --- | ---: | --- |
| Original training CSV | 1,296,675 | 2019-01-01 to 2020-06-21 |
| Untouched future test CSV | 555,719 | 2020-06-21 to 2020-12-31 |

The training dataset contains 983 cardholders. The target `is_fraud` is 0 for
legitimate transactions and 1 for fraud. The original training period is split
chronologically into 80% fitting data and 20% validation data. Raw Kaggle datasets
are not stored in this repository and must not be committed.

Notebook feature engineering includes transaction timing, age at transaction time,
Haversine distance, previous transaction timing, previous average spending, amount
relative to previous spending, previous-hour transaction count, and history flags.
Historical calculations must use only previously available transactions, without
target or temporal leakage. Phase 1 accepts already-engineered inputs; it does not
generate historical features or impute them.

## Models evaluated and final selection

The notebook evaluated Logistic Regression, Random Forest, baseline and tuned
XGBoost, a feedforward neural network, an LSTM with five-transaction sequences,
and a hybrid LSTM/XGBoost system. Tuned standalone XGBoost was selected for its
strongest final precision, F1, ROC-AUC, and PR-AUC, fewer false positives, and full
test coverage without the LSTM's sequence-availability requirement.

Selected parameters: 200 estimators, depth 8, learning rate 0.05, subsample 1.0,
column subsample 0.8, minimum child weight 1. Its saved sklearn Pipeline contains
fitted `OneHotEncoder(handle_unknown="ignore")`, numerical/binary passthrough,
and the fitted XGBoost classifier. It contains no raw feature engineering or
historical imputation. The Logistic Regression/FNN StandardScaler is not production
XGBoost preprocessing.

## Production input contract

The canonical ordered feature definition lives in `src/fraudshield/config.py`:

1. `amt`
2. `category`
3. `gender`
4. `state`
5. `city_pop`
6. `age`
7. `hour`
8. `day_of_week`
9. `month`
10. `is_weekend`
11. `cardholder_merchant_distance_km`
12. `time_since_prev_txn_min`
13. `customer_prev_avg_amt`
14. `amt_ratio_to_prev_avg`
15. `txn_count_prev_1h`
16. `no_prior_history`

`cc_num` may group historical records but is not a predictive input. `is_fraud`
must never be a model input. Reordered columns are accepted and normalized to the
canonical order; the final frame must contain exactly the expected named columns.
Required values cannot be missing, null, NaN, or blank. Numerical values must be
finite numbers; numeric strings are not silently coerced. Binary values are numeric
0 or 1. Other domain ranges and cross-field policies are not invented in this phase.
Unknown nonblank categorical values remain supported. The UI reads category choices
from the fitted encoder, avoiding the prototype's incomplete list.

## Approved test baseline

The production threshold is **0.85**. The inclusive rule `score >= threshold`
produces a fraud alert. Threshold selection used validation data before the final
evaluation on future test transactions.

| Metric | Approved notebook result |
| --- | ---: |
| Fraud precision | approximately 0.7122 |
| Fraud recall | approximately 0.8872 |
| Fraud F1 | approximately 0.7901 |
| ROC-AUC | approximately 0.9981 |
| PR-AUC (average precision) | approximately 0.9058 |
| Accuracy | approximately 99.82% |
| True negatives | 552,805 |
| False positives | 769 |
| False negatives | 242 |
| True positives | 1,903 |
| Model test coverage | 100% |

These are recorded results, not a new local validation. Investigate unexpected
regression differences rather than updating expected values. This baseline measures
the binary model at its threshold, not OOD manual-review routing performance.

## Architecture and guardrails

Streamlit collects engineered inputs → value/schema validation → existing fitted
pipeline → score and threshold classification → application decision with OOD review
precedence. The binary model prediction is retained separately from that decision.
Failed input validation or in-range inference produces an error, never an approval.
An entirely OOD batch can retain manual review with an unavailable score if inference
fails; mixed/in-range batches fail safely without fabricated scores or labels.

Existing notebook/application guardrails trigger strictly above:

| Feature | Limit |
| --- | ---: |
| `amt` | 28,948.90 |
| `txn_count_prev_1h` | 6 |
| `amt_ratio_to_prev_avg` | 676.39 |
| `time_since_prev_txn_min` | 22,357.85 |

These values were observed over the full original training CSV, not demonstrated
as exclusive estimator-fit ranges or statistically validated OOD thresholds. OOD
does not mean fraud. A score may still be calculated, but warnings cause the
application to return **MANUAL REVIEW**, overriding either binary decision.

## AI/ML QA strategy and findings

Independent PyTest tests exercise production functions: the canonical schema,
mandatory fields, NaN, numerical types, schema validation, OOD boundaries,
threshold boundaries, score validity, and artifact/configuration errors.
Mocks isolate software inference boundaries only; they are not trained models and
do not validate production accuracy. Real-artifact tests cover score generation,
unknown categories, reordered columns, and reload consistency. They skip explicitly
when original assets are unavailable; incompatible supplied assets fail.

Notebook findings:

- An overwritten test DataFrame caused regression mismatch; restoring the original
  data reproduced the approved confusion matrix. Future full regression must verify
  dataset identity and train/test temporal separation.
- Missing columns and malformed numeric text were rejected by the saved pipeline,
  but NaN amount values were accepted. Application validation must reject them.
- Unknown categories and reordered columns were handled successfully.
- Reloads produced identical scores; the inclusive threshold boundary was correct.
- Extreme inputs may receive low scores, motivating manual-review guardrails.
- An individual fraud example scored below the threshold: a measured false negative,
  not automatically a software defect.

The preserved notebook contains an earlier training-timestamp assignment to `test_df`
corrected by reloading the test CSV, and a stale QA summary describing QA Test 3 as
pending despite its later corrected rerun. These are not final production behavior.

## Repository structure

```text
FraudShield-AI/
├── AGENTS.md
├── FraudShield_Final_Completed.ipynb
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── app.py
├── src/fraudshield/
│   ├── __init__.py
│   ├── config.py
│   ├── validation.py
│   ├── model_loader.py
│   ├── guardrails.py
│   └── prediction.py
└── tests/
    ├── conftest.py
    ├── test_validation.py
    ├── test_guardrails.py
    ├── test_prediction.py
    └── test_model_loading.py
```

## Installation and usage

Use Python 3.10 or later in an isolated environment:

```shell
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m pytest
python -m streamlit run app.py
```

Production artifacts must be supplied separately until storage strategy is finalized.
Create an ignored local `models/` directory and supply the original trusted
`fraudshield_xgboost_final.pkl` and `xgb_config.json` saved by the notebook. The JSON
must retain its recorded model name and threshold. No replacement model or
configuration is automatically generated. Paths resolve relative to the repository
source location, not the launch working directory. Only load trusted joblib artifacts.

Recover original artifact package versions before deployment. The current unpinned
requirements do not establish serialization compatibility or baseline reproducibility.
Missing/incompatible artifacts produce a setup error without a raw user traceback,
retraining, or substitute predictions.

## Current limitations

Original production artifacts are absent, so real inference has not yet been verified
locally. Full test-set regression and integrity checks require approved assets and
are deferred. Verified full-precision history medians are unavailable; no
`history_config.json` is fabricated. Cold-start implementations differ and need an
approved policy before raw historical reconstruction. Users provide every engineered
history input, including the ratio and history flag.

There is no customer-history store, API, continuous monitoring system, or claimed
production deployment. Explainability and deep-learning research remain in the
notebook. No training occurs during application startup or inference.
