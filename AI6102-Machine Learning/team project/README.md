# AI6102 Team Project: Titanic Survival Prediction

**Kaggle competition:** [Titanic – Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
**Course:** AI6102 Machine Learning Methodologies & Applications, NTU (November 2026)
**Team "AI6102 Project":** JI QIANKUN, GAN MINGRUI

- **Report:** [report/report.pdf](report/report.pdf)
- **Video summary (about 3 minutes, with narration):** https://youtu.be/1Qk1QqcjpkA

## Result

Our final model (v7) is CatBoost with each passenger's ticket group and family group as native categorical features. CatBoost's ordered target statistics let the model learn from travelling companions without leaking a passenger's own label. It reaches **84.74%** stratified 5-fold cross-validation accuracy and a Kaggle public score of **0.80382** (336 of 418 test passengers correct).

On 18 November 2026 this was rank 821 of 11,283 teams (top 7.3%). The Titanic answer key is publicly available, and 408 leaderboard entries at 0.85885 or above are consistent with looked-up labels; without them our rank is about 413 of 10,875 (top 3.8%). We used only the official competition files.

| Version | Model | Stratified CV | Kaggle |
|---|---|---:|---:|
| v1 | RF + HGB blend, raw features + Cabin flag | 83.95% | 0.77033 |
| v2 | Logistic regression, full features | 83.61% | 0.76076 |
| v4 | HGB, 7 leaves, no Deck | 84.29% | 0.75119 |
| v5 | Random forest recipe (title, ticket prefix) | 82.94% | 0.77990 |
| v6 | v5 + cross-fitted co-traveller target encoding | 84.29% | 0.79665 |
| **v7** | **CatBoost, native ticket/family groups** | **84.74%** | **0.80382** |
| v12 | Role-group blend (exploratory) | 85.02% | 0.79186 |

The report covers all 15 experiment rounds (about 2,050 cross-validation fits), including the ablations and the rounds that did not help.

## Files

```
team project/
├── titanic_v7.py           Final model: features, cross-validation, submission (single file)
├── requirements.txt
├── report/                 Final report: PDF, LaTeX source, figures
└── results/
    ├── fold_assignments.csv    Fixed 5-fold splits used in every experiment
    ├── v7_cv_summary.csv       CV accuracy of all v7 candidates (stratified, fresh-seed, ticket-group)
    └── v7_submission.csv       The submitted file (Kaggle 0.80382)
```

`titanic_v7.py` is a standalone version of the submitted model. Its feature code is copied unchanged from the project modules and gives identical feature tables. The full experiment code for all versions (v1–v13) is kept in our project archive and is not included here to keep the repository small.

## Run it

Tested with Python 3.11.

1. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Download the data from the [Kaggle data page](https://www.kaggle.com/competitions/titanic/data) (accept the competition rules first) and put `train.csv` and `test.csv` in a new `data/` folder:
   ```
   team project/data/train.csv
   team project/data/test.csv
   ```
3. Run:
   ```bash
   python titanic_v7.py           # cross-validation (10 fits, a few minutes) + submission.csv
   python titanic_v7.py --no-cv   # only train on all data and write submission.csv
   ```

The script prints the fold accuracies, compares them with the report (84.74% stratified, 81.93% ticket-group), and writes `submission.csv`.

## Method in brief

- **Features:** title from the name (Mr, Mrs, Miss, Master, Officer, Royalty), deck, ticket prefix, family size and its bands, plus two group keys: the normalised ticket and surname + family size.
- **No leakage:** age, fare and embarkation are imputed from the training rows only; group statistics come from CatBoost's ordered target statistics, which never use a row's own label.
- **Model:** CatBoost, depth 4, 600 iterations, learning rate 0.035, L2 = 5, ordered boosting, seed 42, threshold 0.5, no early stopping.
- **Validation:** fixed stratified 5-fold CV plus ticket-group 5-fold CV, which checks performance on passengers whose companions are unseen.
