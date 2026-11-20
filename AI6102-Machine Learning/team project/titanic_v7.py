"""
AI6102 Titanic team project -- final model (v7), standalone version.

CatBoost with the passenger's ticket group and family group as native
categorical features. CatBoost's ordered target statistics encode each group
using only rows that come earlier in a random permutation, so a passenger's own
label never leaks into its own feature.

Usage
    python titanic_v7.py           # cross-validation + final fit + submission.csv
    python titanic_v7.py --no-cv   # final fit + submission.csv only

Input   data/train.csv, data/test.csv (download from Kaggle, see README)
Output  submission.csv (PassengerId, Survived)

The feature code below is copied unchanged from the project modules used for
the submitted model (reference_recipe.py, relationship_features.py,
baseline.py, catboost_v7.py). All statistics are learned from the rows passed
to fit(), so cross-validation folds never see their validation rows.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# Fixed v7 settings (no early stopping, no threshold search; threshold 0.5)
PARAMS = dict(iterations=600, learning_rate=0.035, l2_leaf_reg=5.0, depth=4,
              loss_function="Logloss", boosting_type="Ordered", counter_calc_method="SkipTest",
              random_seed=42, thread_count=2, verbose=False, allow_writing_files=False)

TITLE_MAP = {**dict.fromkeys(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'),
             **dict.fromkeys(['Jonkheer', 'Don', 'Sir', 'the Countess', 'Lady', 'Dona'], 'Royalty'),
             'Mme': 'Mrs', 'Mlle': 'Miss', 'Ms': 'Mrs',
             **{v: v for v in ['Mr', 'Mrs', 'Miss', 'Master']}}
NUMERIC = ['Age', 'Fare', 'SibSp', 'Parch', 'SexCode', 'FamilySize', 'Singleton', 'SmallFamily', 'LargeFamily']
CATEGORICAL = ['Pclass', 'Embarked', 'Deck', 'Title', 'TicketPrefix']
GROUPS = ['TicketGroup', 'FamilyGroupKey']
CAT_FEATURES = CATEGORICAL + GROUPS


# ------------------------------------------------------------------ features
def raw_features(frame):
    result = frame[['Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'Embarked', 'Sex']].copy()
    title = frame.Name.astype('string').str.extract(r',\s*([^.]*)\.', expand=False).str.strip()
    result['Title'] = title.map(TITLE_MAP).fillna('Unknown').astype(str)
    result['Deck'] = frame.Cabin.astype('string').str.strip().str[0].fillna('U').replace('', 'U').astype(str)
    ticket = frame.Ticket.astype('string').fillna('').str.replace('.', '', regex=False).str.replace('/', '', regex=False)
    result['TicketPrefix'] = ticket.map(lambda value: next((part for part in value.split() if not part.isdigit()), 'XXX'))
    result['SexCode'] = frame.Sex.map({'male': 1., 'female': 0.})
    size = frame.SibSp + frame.Parch + 1
    result['FamilySize'] = size
    result['Singleton'] = size.eq(1).astype(float)
    result['SmallFamily'] = size.between(2, 4).astype(float)
    result['LargeFamily'] = size.ge(5).astype(float)
    return result


def ticket_groups(frame):
    tickets = frame['Ticket'].astype('string').str.strip().str.upper().str.replace(r'\s+', ' ', regex=True)
    missing = tickets.isna() | tickets.eq('')
    tickets = 'ticket:' + tickets
    tickets.loc[missing] = 'missing_id:' + frame.loc[missing, 'PassengerId'].astype(str)
    return tickets.astype(str).to_numpy()


def group_keys(frame):
    surname = frame.Name.astype('string').str.split(',').str[0].str.strip().str.upper()
    size = frame.SibSp + frame.Parch + 1
    family = surname + '|size:' + size.astype('Int64').astype('string')
    valid = surname.notna() & surname.ne('') & size.gt(1)
    family = family.where(valid, 'singleton:' + frame.PassengerId.astype(str))
    return ticket_groups(frame), family.astype(str).to_numpy()


class Imputer:
    """Age by (Sex, Pclass, Title) median, then (Sex, Pclass), then global; Fare by mean; Embarked 'S'."""

    def fit(self, X):
        frame = raw_features(X)
        self.age_median_ = float(frame.Age.median())
        self.fare_mean_ = float(frame.Fare.mean())
        self.group_medians_ = frame.groupby(['Sex', 'Pclass', 'Title']).Age.median().dropna().to_dict()
        self.class_medians_ = frame.groupby(['Sex', 'Pclass']).Age.median().dropna().to_dict()
        if not np.isfinite([self.age_median_, self.fare_mean_]).all():
            raise ValueError('Age and Fare require nonmissing training data')
        return self

    def transform(self, X):
        frame = raw_features(X)
        fills = [self.group_medians_.get((sex, pclass, title), self.class_medians_.get((sex, pclass), self.age_median_))
                 for sex, pclass, title in zip(frame.Sex, frame.Pclass, frame.Title)]
        frame['Age'] = frame.Age.fillna(pd.Series(fills, index=frame.index))
        frame['Fare'] = frame.Fare.fillna(self.fare_mean_)
        frame['Embarked'] = frame.Embarked.fillna('S')
        return frame[NUMERIC + CATEGORICAL]


def make_features(imputer, X):
    base = imputer.transform(X)
    base['TicketGroup'], base['FamilyGroupKey'] = group_keys(X)
    for column in CAT_FEATURES:
        if column == 'Pclass':
            base[column] = base[column].astype('Int64').astype('string')
        base[column] = base[column].astype('string').fillna('__MISSING__').astype(str)
    return base


# ------------------------------------------------------------------ model
def fit(X, y):
    from catboost import CatBoostClassifier
    imputer = Imputer().fit(X)
    model = CatBoostClassifier(**PARAMS)
    model.fit(make_features(imputer, X), y, cat_features=CAT_FEATURES)
    return imputer, model


def predict_proba(fitted, X):
    imputer, model = fitted
    return model.predict_proba(make_features(imputer, X))[:, 1]


def cross_validate(X, y):
    """Fixed folds used throughout the project (results/fold_assignments.csv)."""
    folds = pd.read_csv(ROOT / 'results' / 'fold_assignments.csv').set_index('PassengerId').loc[X.PassengerId]
    for scheme, label in [('stratified_5fold', 'stratified 5-fold'), ('ticket_group_5fold', 'ticket-group 5-fold')]:
        accuracies = []
        for k in range(1, 6):
            valid = folds[scheme].to_numpy() == k
            p = predict_proba(fit(X[~valid], y[~valid]), X[valid])
            accuracies.append(float(((p > .5).astype(int) == y[valid]).mean()))
            print(f'  {label}: fold {k}/5 accuracy {accuracies[-1]:.4f}', flush=True)
        print(f'{label} mean accuracy: {np.mean(accuracies) * 100:.2f}%  (report: '
              f'{"84.74" if scheme == "stratified_5fold" else "81.93"}%)\n', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('--no-cv', action='store_true', help='skip cross-validation')
    parser.add_argument('--out', default='submission.csv', help='output file name')
    args = parser.parse_args()

    train = pd.read_csv(ROOT / 'data' / 'train.csv')
    test = pd.read_csv(ROOT / 'data' / 'test.csv')
    X, y = train.drop(columns='Survived'), train.Survived.to_numpy()

    if not args.no_cv:
        cross_validate(X, y)
    p = predict_proba(fit(X, y), test)
    submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': (p > .5).astype(int)})
    submission.to_csv(ROOT / args.out, index=False)
    print(f'Wrote {args.out}: {len(submission)} rows, {int(submission.Survived.sum())} predicted survivors '
          f'(submitted v7 file: 146; Kaggle 0.80382)')


if __name__ == '__main__':
    main()
