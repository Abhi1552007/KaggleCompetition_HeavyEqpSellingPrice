# Heavy Equipment Selling Price Prediction

My solution to the Kaggle **Heavy Equipment Selling Price Prediction Challenge**: predicting the resale price of used construction machinery (excavators, loaders, dozers …) from sale records and machine specifications.

## Results

Validation RMSLE (lower is better):

| Model | RMSLE |
|---|---|
| Mean-price baseline | 0.466 |
| Linear Regression | 0.426 |
| Random Forest | 0.291 |
| XGBoost (5-fold) | 0.217 |
| **LightGBM, 5-fold ensemble** | **0.205** |

## Approach

1. **Target transform.** Prices are heavily right-skewed, so the model predicts `log(1 + price)`, which gives a near-normal target and aligns with the RMSLE metric.
2. **EDA findings that shaped the features**
   - Manufacture year `1001` is a placeholder for "unknown" (≈14.8k rows), handled explicitly rather than treated as a real year.
   - Many "missing" spec columns are structurally missing, not noise: `col4` exists only for backhoe loaders and `col18` only for skid steer loaders.
   - Track excavators make up over half the data; machine type and size drive most of the price variation.
   - Region matters modestly; the extreme regional averages come from tiny samples (e.g. Washington DC has only 2 sales).
3. **Feature engineering:** asset age at sale, sale month, hours per year of use, spec completeness, spec subclass, utilisation tier and cabin type.
4. **Modelling:** linear and random-forest baselines, then LightGBM tuned with `RandomizedSearchCV` and trained as a 5-fold K-Fold ensemble, with XGBoost as a comparison.

## Running it

The notebook is written for Kaggle. Attach the competition dataset and run all cells in
[`25f2001101-notebook-2026t2.ipynb`](25f2001101-notebook-2026t2.ipynb). Locally you'll need `pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn`, with the data paths updated.
