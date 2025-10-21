import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    x1 = np.array(x1)
    x2 = np.array(x2)

    dist = np.sum((x1 - x2)**2)**0.5

    return dist


class KNNclassifier:
    def __init__(self, k = 3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)


    

    def _predict(self, x):
        distance = []
        for xi in self.X_train:
            distance.append(euclidean_distance(x, xi))

        k_indices =  np.argsort(distance)[:self.k]

        k_nearest_neighbors = self.y_train[k_indices]

        most_common = Counter(k_nearest_neighbors).most_common(1)
        # most_common would be [(label, count)]
        return most_common[0][0]


    def predict(self, X_test):
        predictions = []
        for x in X_test:
            predictions.append(self._predict(x))
        return predictions
            

    
    