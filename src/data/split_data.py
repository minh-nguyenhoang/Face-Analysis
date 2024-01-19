import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

df = pd.read_csv('src/data/label_cropped.csv', index_col=0)

df.loc[:, 'fold'] = -1 # Create a new column `fold` containing `-1`s.
df = df.sample(frac=1).reset_index(drop=True) # Shuffle the rows.
targets = df.drop('file_name', axis=1).values # Extract the targets as an array.

mskf = MultilabelStratifiedKFold(n_splits=5)

for fold_, (train_, valid_) in enumerate(mskf.split(X=df, y=targets)):
    df.loc[valid_, 'fold'] = fold_
    
df.to_csv('./targets_with_folds.csv', index=False)