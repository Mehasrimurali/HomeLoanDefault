# Home Loan Default Prediction

## 1. Overview
An end-to-end credit-risk project that predicts which loan applicants are likely to **default**, and which customer segments are safest to approve. It joins several relational tables into one applicant-level dataset, compares six classifiers on a heavily imbalanced target, and turns the chosen model into **risk bands** a lender can use.

## 2. Objectives
- **Task 1:** analyse the data and report what drives default risk.
- **Task 2:** build a predictive model that flags risky applicants and identifies customer groups that are low risk.
- Compare models, recommend one for production, and report the challenges met and how they were handled.

## 3. Dataset
Home Credit style data from the PRCP-1006 capstone brief (`Data/` folder, not included in this repository):

| Table | Used for |
|---|---|
| `application_train.csv` | One row per applicant: 307,511 rows, 122 columns, target `TARGET` (1 = default) |
| `bureau.csv`, `bureau_balance.csv` | Credit history from other institutions |
| `previous_application.csv` | Earlier loan applications at the lender |
| `POS_CASH_balance.csv` | Monthly point-of-sale and cash loan history |
| `credit_card_balance.csv` | Monthly credit card history |

The target is imbalanced: about **8% of applicants default**, so a model that predicts "no default" for everyone is 91.9% accurate but useless.

## 4. Technologies
Python, Pandas, NumPy, SciPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, Jupyter Notebook.

## 5. Project Workflow
Load tables (chunked, with memory reduction) → data checks and cleaning → aggregate each child table to one row per applicant → engineered ratio features (125 final features) → stratified train/validation/test split → preprocessing inside pipelines → class-weighted training of six models plus a baseline → cross-validation → tuning → threshold chosen on validation → single test evaluation → risk bands, feature importance, error analysis → challenges report.

## 6. Key Design Choices
- **No leakage:** `TARGET` is excluded from features, the test file is not used, preprocessing is fit only inside training pipelines, and the decision threshold is chosen on the **validation** set and applied **once** to the test set.
- **Imbalance handled** with class weights, and judged with ROC-AUC, PR-AUC, KS statistic and recall on defaulters, not accuracy.
- **Big tables** are read in chunks and aggregated per applicant to keep memory manageable.
- **Tuning is accepted only if it helps:** tuned XGBoost did not improve validation ROC-AUC, so the default model was kept.

## 7. Model Comparison (validation set)
| Model | ROC-AUC | PR-AUC | KS | Recall (defaulters) |
|---|---|---|---|---|
| **HistGradientBoosting** | **0.7726** | 0.2679 | 0.4123 | 0.675 |
| XGBoost | 0.7714 | 0.2638 | 0.4027 | 0.644 |
| Logistic Regression | 0.7576 | 0.2423 | 0.3838 | 0.687 |
| Random Forest | 0.7541 | 0.2389 | 0.3780 | 0.591 |
| Extra Trees | 0.7470 | 0.2314 | 0.3656 | 0.632 |
| Decision Tree | 0.7213 | 0.2045 | 0.3336 | 0.649 |
| Dummy baseline | 0.5000 | 0.0807 | 0.0000 | 0.000 |

## 8. Final Test Results
Selected model: **HistGradientBoosting**, threshold 0.505 (chosen on validation).

| Metric | Test |
|---|---|
| ROC-AUC | 0.7773 |
| PR-AUC | 0.2660 |
| KS statistic | 0.4161 |
| Recall on defaulters | 0.6745 |
| Precision on defaulters | 0.1854 |
| F1 on defaulters | 0.2909 |
| Brier score | 0.1806 |

Precision is low because catching two thirds of defaulters means also flagging many safe customers. That is the usual trade-off on this kind of data and depends on the lender's costs.

## 9. Customer Risk Bands (test set)
| Risk band | Customers | Actual default rate | Share of customers |
|---|---|---|---|
| Low | 14,771 | 1.82% | 32.0% |
| Medium | 17,807 | 5.30% | 38.6% |
| High | 13,549 | 18.54% | 29.4% |

The high-risk group defaults about **10 times more often** than the low-risk group, so the bands separate applicants well.

## 10. What Drives Default Risk
Permutation importance shows the external credit scores (average of `EXT_SOURCE` fields) as by far the strongest factor, followed by credit term, gender, age, employment length, goods-to-credit ratio, education and annuity amount.

## 11. Limitations
- Moderate ROC-AUC (about 0.78): the model ranks risk usefully but is not highly precise.
- `installments_payments.csv` was not available, so repayment-timing features are missing.
- Gender and age appear among important features. A real lender would need a fairness and regulatory review before using them.

## 12. How to Run
1. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the dataset from the course brief and place the CSV files in a `Data/` folder next to the notebook (subfolders are searched).
3. Run the notebook (the full run takes a while because of the large tables):
   ```bash
   jupyter notebook PRCP-1006-HomeLoanDef_Final.ipynb
   ```
   The last cell is an audit that checks the main rules of the brief.

## 13. Repository Structure
```
home-loan-default-prediction/
├── Data/                  (not included, see "How to Run")
├── PRCP-1006-HomeLoanDef_Final.ipynb
├── README.md
└── requirements.txt
```
