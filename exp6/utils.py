import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    split_point = int(n_samples * (1 - test_size))
    
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    X_train = X[train_indices].copy()
    X_test = X[test_indices].copy()
    y_train = y[train_indices].copy()
    y_test = y[test_indices].copy()

    return X_train, X_test, y_train, y_test
