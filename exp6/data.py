import pandas as pd
from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id = 53)
# dataset = fetch_ucirepo(id = 109) # for wine dataset
X = dataset.data.features
y = dataset.data.targets

# print(y.head())

y = y.apply(lambda col: col.str.replace("Iris-", "")) #comment this line for wine dataset

# print(y.head())

X_np = X.to_numpy()
y_np = y.to_numpy().ravel()