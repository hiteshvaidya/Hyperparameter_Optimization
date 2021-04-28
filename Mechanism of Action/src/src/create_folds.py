import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == '__main__':
    df = pd.read_csv('../data/train_targets_scored.csv.csv')
    df.loc[:, 'kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)

    targets = df.drop('sig_id', axis=1).values

    mskf = MultilabelStratifiedKFold