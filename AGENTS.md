# FraudShield AI — Project Context and Engineering Instructions

## Project Purpose

FraudShield AI is an end-to-end fraud detection portfolio project designed to demonstrate:

- Machine learning model development
- Deep learning experimentation
- Feature engineering
- Imbalanced classification
- Model evaluation
- Production inference
- Streamlit deployment
- AI/ML quality assurance and testing
- Production validation and guardrails

The project is intended to demonstrate both ML engineering and AI Test / AI Quality Engineering skills.

The completed notebook:
`FraudShield_Final_Completed.ipynb`

must be treated as the primary source of truth for the work already completed.

Do not modify the completed notebook unless explicitly instructed.

---

## Dataset

Kaggle Credit Card Fraud dataset.

Training dataset:
- Approximately 1.3 million transactions
- 983 cardholders
- Date range:
  2019-01-01 through 2020-06-21

Test dataset:
- 555,719 transactions
- Date range:
  2020-06-21 through 2020-12-31

Target:
`is_fraud`

- 0 = legitimate transaction
- 1 = fraudulent transaction

Raw datasets should NOT be committed to GitHub.

---

## Final Production Model

Primary production model:

Tuned XGBoost classifier.

Do not retrain or replace the production model unless explicitly requested.

Production decision threshold:

0.85

Decision rule:

Fraud risk score >= 0.85
→ Fraud Alert

Fraud risk score < 0.85
→ No Fraud Alert

The model score should preferably be described as a
"Fraud Risk Score" rather than as a perfectly calibrated real-world probability.

---

## Final Production Features

The production model expects exactly these 16 model features:

1. amt
2. category
3. gender
4. state
5. city_pop
6. age
7. hour
8. day_of_week
9. month
10. is_weekend
11. cardholder_merchant_distance_km
12. time_since_prev_txn_min
13. customer_prev_avg_amt
14. amt_ratio_to_prev_avg
15. txn_count_prev_1h
16. no_prior_history

Do not add, remove, rename, or reorder the logical feature definition without explicit approval.

`cc_num` may be used as a grouping/system identifier for historical feature engineering but must NOT be passed into the final predictive model.

`is_fraud` is the target and must never be included as a model input.

---

## Feature Groups

Numerical features:

- amt
- city_pop
- age
- hour
- month
- cardholder_merchant_distance_km
- time_since_prev_txn_min
- customer_prev_avg_amt
- amt_ratio_to_prev_avg
- txn_count_prev_1h

Binary features:

- is_weekend
- no_prior_history

Categorical features:

- category
- gender
- state
- day_of_week

The production sklearn pipeline performs preprocessing and XGBoost inference.

Categorical encoding uses behavior equivalent to:

OneHotEncoder(handle_unknown="ignore")

---

## Production Pipeline Boundary

The saved production XGBoost artifact is an sklearn Pipeline containing:

- fitted ColumnTransformer
- OneHotEncoder(handle_unknown="ignore") for categorical features
- numerical feature passthrough
- binary feature passthrough
- fitted XGBClassifier

The saved pipeline does NOT perform:

- raw transaction feature engineering
- historical feature generation
- historical median imputation

Those operations occur before the final 16-feature DataFrame is passed to the
saved XGBoost pipeline.

Do not use the StandardScaler preprocessing used by Logistic Regression/FNN as
production XGBoost preprocessing.

The production model therefore expects the already-engineered final 16-feature
schema.

---

## Supported Categorical Values

EDA identifies 14 transaction categories.

The Streamlit prototype contains a hard-coded list of only 13 categories and
omits grocery_net.

Do not copy that incomplete list into the final production application.

When the saved fitted XGBoost pipeline is locally available, prefer obtaining
trained categorical values from the fitted OneHotEncoder or another verified
configuration source.

Unknown categories remain technically supported because the encoder uses
handle_unknown="ignore".

Do not automatically classify an unknown category as fraud or manual review
unless a new explicit business policy is approved.

---

## Final Approved XGBoost Test Results

Production threshold: 0.85

Approved untouched test-set confusion matrix:

- True Negatives: 552,805
- False Positives: 769
- False Negatives: 242
- True Positives: 1,903

Metrics:

- Precision: approximately 0.7122
- Recall: approximately 0.8872
- F1-score: approximately 0.7901
- ROC-AUC: approximately 0.9981
- PR-AUC: approximately 0.9058

These values form the regression baseline.

Do not silently replace these values with newly calculated results.
If regression results differ, investigate the cause.

---

## Other Models Developed

The project also explored:

- Logistic Regression
- Random Forest
- XGBoost
- Tuned XGBoost
- Feedforward Neural Network (FNN)
- LSTM
- Hybrid LSTM + XGBoost approach

Final ranking showed tuned XGBoost as the strongest production model.

Deep-learning models should remain represented in the notebook/documentation because they are part of the project experimentation, but they should not replace tuned XGBoost as the primary production model unless explicitly requested.

---

## Important Historical Features

Historical features include:

- time_since_prev_txn_min
- customer_prev_avg_amt
- amt_ratio_to_prev_avg
- txn_count_prev_1h
- no_prior_history

Historical calculations must be causal.

Future transactions must never be used to create features for earlier transactions.

Avoid target leakage and temporal leakage.

---

## History Configuration Limitation

The notebook shows historical-feature imputation values during experimentation,
but a verified full-precision history-median configuration artifact is not
currently available in the repository.

Do NOT:

- guess the historical medians
- round displayed notebook values and treat them as production constants
- create history_config.json containing unverified values

Exact history-imputation values must be recovered from an existing artifact or
recomputed using the approved training procedure before raw transaction history
reconstruction is implemented in production.

Until then, the first production/demo application may accept the already
engineered historical model inputs rather than pretending to reconstruct them
from a production history store.

---

## Cold-Start and no_prior_history Policy

The notebook contains multiple implementations of historical cold-start handling.

Differences exist around:

- how no_prior_history is derived
- which missing historical features trigger it
- zero previous-average behavior
- infinite amount ratios
- missing-value imputation

Do not silently combine these implementations.

Before implementing raw historical feature generation as reusable production
code, establish one explicit policy based on the approved training behavior.

For the initial portfolio application, preserve compatibility with the existing
final 16-feature model input contract.

---

## QA Findings

The notebook contains a dedicated AI/ML QA section.

Important findings include:

### Dataset integrity issue

During QA regression testing, a `test_df` variable had accidentally been overwritten later in the notebook.

This caused regression results to differ significantly from the approved baseline.

After restoring the correct Kaggle test dataset, the exact approved confusion matrix was reproduced.

Production implication:

Add data-integrity validation and avoid reusing ambiguous dataframe names.

Prefer names such as:

- raw_train_df
- raw_test_df
- validation_df
- qa_test

---

### Missing required column

Removing a required feature such as `amt` caused the production sklearn pipeline to reject the input.

Production behavior should still validate schema before calling the model rather than relying solely on sklearn exceptions.

---

### Invalid numerical type

Setting:

amt = "abc"

caused the production preprocessing pipeline to reject the input.

Application validation should catch invalid types before model inference.

---

### Missing / NaN values

Setting a mandatory numeric field such as:

amt = NaN

did NOT necessarily cause XGBoost to fail.

XGBoost can technically generate predictions with missing values.

However, transaction amount is a mandatory FraudShield business field.

Therefore:

Mandatory required fields must be explicitly checked for null / NaN values before model inference.

Model acceptance of NaN does not mean the input is valid from a business perspective.

---

### Unknown categorical values

Previously unseen category values can be processed because the categorical preprocessing uses:

handle_unknown="ignore"

The system should not crash on a new category.

However, successfully producing a score does not prove that prediction reliability is equivalent to known categories.

---

### Column order

The pipeline produced the same prediction when the 16 input DataFrame columns were reordered.

The pipeline identifies features using column names rather than relying on physical column position.

---

### Threshold boundary

Production threshold logic is:

score < 0.85 → prediction 0
score = 0.85 → prediction 1
score > 0.85 → prediction 1

The production logic must use:

score >= threshold

---

### Model reload consistency

Repeated loading of the same saved XGBoost model and passing the same input produced identical fraud scores and classifications.

---

## OOD Guardrail Design

Some model inputs may be far outside values observed in the full original
training CSV.

The currently documented guardrail limits are values observed over the full
original training CSV.

They were not demonstrated to be limits calculated exclusively from the
estimator's first 80% fitting subset.

Preserve these values as the existing notebook/application guardrail policy:

- amt > 28948.90
- txn_count_prev_1h > 6
- amt_ratio_to_prev_avg > 676.39
- time_since_prev_txn_min > 22357.85

Use strict greater-than comparisons as demonstrated in the notebook.

Do not describe these values as statistically validated OOD thresholds or as
exclusive estimator-fit ranges.

The application should support guardrails for configured out-of-distribution inputs.

Important:

OOD does NOT automatically mean fraud.

Current intended application behavior remains:

OOD input
→ model score may still be calculated
→ warning is generated
→ application decision becomes MANUAL REVIEW

Do not silently convert OOD rules into fraud labels.

---

## Production Validation Layers

FraudShield should use multiple validation layers.

### Layer 1 — Application/Input Validation

Before model inference validate:

- required fields
- null / NaN values
- correct data types
- valid supported input ranges where appropriate
- supported UI selections

Invalid user input should be rejected before reaching the model.

### Layer 2 — Model Schema Validation

Immediately before inference verify that the final model DataFrame contains the expected schema.

This protects against:

- API integrations
- batch jobs
- alternate frontends
- transformation bugs
- developer scripts
- accidental dropped columns

### Layer 3 — Model Inference

Only validated inputs should reach:

predict_proba()

Model/pipeline exceptions should be handled gracefully rather than displaying raw stack traces to application users.

---

## Streamlit Application

The current project included Streamlit prototype work inside the development workflow.

The final GitHub project should contain a clean production-style `app.py`.

The app should:

- load the saved XGBoost model
- load configuration
- collect or construct all required model features
- validate mandatory fields
- reject NaN mandatory fields
- validate schema
- apply OOD guardrails
- calculate the fraud risk score
- apply the 0.85 production threshold
- display a clear fraud/no-fraud-alert decision
- show manual-review messaging for OOD conditions
- handle errors gracefully

Do not retrain models inside Streamlit.

---

## Testing Strategy

Important notebook QA checks should eventually become independent PyTest tests.

Tests should test production functions rather than reimplementing the same logic inside the test.

Important automated tests should cover:

- model file loading
- configuration loading
- prediction score range
- production threshold behavior
- threshold boundary values
- missing required columns
- required-field validation
- NaN validation
- invalid numerical types
- unseen categorical values
- model schema validation
- column-order robustness
- model reload consistency
- regression baseline
- OOD guardrail behavior
- data integrity
- train/test temporal separation

Where appropriate, tests should import actual production functions from `src/`.

---

## Important QA Philosophy

Do not confuse model errors with software defects.

Example:

A real fraud transaction may receive a score below 0.85 and become a false negative.

That is a measured model-performance limitation, not automatically a software bug.

Regression testing should detect unexpected changes in system behavior.

If approved metrics change unexpectedly:

Do not simply update the expected values.

Investigate:

- dataset identity
- preprocessing
- feature engineering
- model version
- threshold
- configuration
- schema
- data leakage

---

## Known Notebook Inconsistencies

The completed notebook is intentionally preserved as research provenance and
contains two known narrative/code inconsistencies:

1. An earlier notebook cell temporarily assigns training timestamps to test_df
   before the proper test CSV is subsequently loaded.

2. The written QA summary still states that QA Test 3 requires rerunning,
   although later notebook cells contain the completed corrected rerun.

These issues must not be treated as the final production behavior.

Do not modify the completed notebook simply to remove these historical artifacts.
Account for them when extracting production code and documentation.

---

## GitHub Refactoring Goal

Convert the notebook project into a clean production-style repository while preserving the notebook as evidence of the full ML development process.

A likely structure is:

FraudShield-AI/
├── AGENTS.md
├── README.md
├── requirements.txt
├── app.py
├── notebooks/
├── src/
├── tests/
├── models/
├── config/
├── docs/
└── assets/

Exact structure may be refined after notebook inspection.

---

## Repository Principles

1. Preserve the completed notebook.
2. Do not commit raw Kaggle datasets.
3. Do not retrain models unnecessarily.
4. Keep production inference separate from experimentation.
5. Keep reusable code in `src/`.
6. Keep automated QA in `tests/`.
7. Avoid duplicated model thresholds and feature definitions.
8. Centralize production configuration.
9. Prefer readable code suitable for a portfolio and technical interview.
10. Do not invent functionality unsupported by the notebook.
11. If notebook evidence conflicts with this file, flag the discrepancy before changing production logic.
12. Ask for clarification rather than silently guessing.

---
