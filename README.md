# Music Streaming Churn Prediction

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.96-success.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Top%205-orange.svg)

## Overview

ML system predicting user churn for a music streaming platform using LightGBM on behavioral event logs.

**Kaggle Competition Result: 5th place**

## Results

**ROC-AUC: 0.96** (10-fold CV)

Cross-validation scores: 0.9512 - 0.9676 (low variance = robust model)

## Challenge

Predict if users will cancel within 10 days using:
- 17.5M events, 19,140 users
- Variable-length event sequences
- Imbalanced classes
- Temporal data with leakage risk

## Approach

**Feature Engineering (50+ features):**
- Activity: sessions, songs played, recency
- Engagement: thumbs up/down, playlist additions
- Temporal: activity trends, session patterns
- Problems: errors, help visits, downgrades
- Interactions: cancel × inactivity, frustration scores

**Model:**
- LightGBM with class balancing
- Temporal cutoffs prevent leakage
- Stratified 10-fold validation

**Top Predictors:**
1. Cancel page visits
2. Downgrade attempts  
3. Days since last activity
4. Error rates
5. Activity decline

## Technologies

Python, LightGBM, Pandas, NumPy, Scikit-learn

## Quick Start
```bash
pip install pandas numpy scikit-learn lightgbm pyarrow
jupyter notebook notebooks/churn_prediction.ipynb
```

## Business Impact

- Identify at-risk users before churn
- Target retention campaigns
- Reduce acquisition costs
- ROC-AUC 0.96 >> industry standard (0.7-0.85)

## License

MIT License

## Author

**Tom Weinberg** - [@tomwbrg](https://github.com/tomwbrg)
