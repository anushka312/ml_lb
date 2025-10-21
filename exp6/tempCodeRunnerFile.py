import pandas as pd
from ucimlrepo import fetch_ucirepo


iris_dataset = fetch_ucirepo(id = 53)

X = iris_dataset.data.features
y = iris_dataset.data.targets