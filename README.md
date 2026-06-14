# CEI Data Science Assignments
**Student:** Ananya Garg

---

## Week 1 — Python & ML Fundamentals
**File:** `week1_Ananya_Garg.ipynb`

Covers Python basics, NumPy, Pandas, Linear Algebra, Statistics, and Probability.

---

## Week 2 — End-to-End ML Pipeline
**File:** `week2_Ananya_Garg.ipynb`  
**Dataset:** Tesla EA Deliveries & Production (2015–2025) — 2,640 rows × 12 columns
**Target:** `Estimated_Deliveries`, `Avg_Price_USD`

### Pipeline Structure
| Step | Description |
|------|-------------|
| Preprocessing | Null/duplicate check, dropped metadata column |
| EDA | Yearly trends, model/region breakdown, price distribution, correlation heatmap |
| Feature Engineering | Date ordinal, One-Hot Encoding for Region & Model |
| Regression Modeling | Ridge, Lasso, Random Forest, Gradient Boosting |
| Cross-Validation | 5-fold CV on best model |
| Hyperparameter Tuning | GridSearchCV on Gradient Boosting |
| Time Series Forecasting | Holt-Winters vs SARIMA — future 4Q forecast |
| Price Prediction | Separate GBM model predicting Avg_Price_USD |

---

*More assignments coming soon (Weeks 3–8)*
