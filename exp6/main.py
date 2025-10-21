import numpy as np
from utils import train_test_split
from data import X_np, y_np
from knn_classifier import KNNclassifier
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = train_test_split(X_np, y_np, 0.2, 42)

model = KNNclassifier(3)

model.fit(X_train=X_train, y_train=y_train)

y_pred = model.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

num_test_samples = len(y_test)
num_correct_predictions = np.sum(y_pred == y_test)
print(f"Number of test samples: {num_test_samples}")
print(f"Number of correct predictions: {num_correct_predictions}")


k_values = [1, 3, 5, 7, 9, 11, 15]

accuracies = []

for k in k_values:
    model = KNNclassifier(k=k)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    acc = np.mean(y_pred == y_test)
    accuracies.append(acc)
    
    print(f"k={k}, Accuracy={acc*100:.2f}%")


plt.figure(figsize=(8,5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='red')
plt.xlabel("k-value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy vs. k-value")
plt.xticks(k_values) 
plt.grid(True)
plt.show()
