import pandas as pd
import numpy as np

df = pd.read_csv('src/data/label_cropped.csv', index_col=0)
print(df.head())
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

train = df[msk].reset_index()
test = df[~msk].reset_index()

train = train.drop(columns=['index'])
test = test.drop(columns=['index'])
print(train.head())
train.to_csv('train.csv',index=False)
test.to_csv('test.csv',index=False)