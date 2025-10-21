import matplotlib.pyplot as plt
import pandas as pd
from data import X_np, y_np
import numpy as np

#comment this section for wine dataset
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'] 
X_df = pd.DataFrame(X_np, columns=feature_names)

classes, y_numbers = np.unique(y_np, return_inverse=True)

for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        plt.figure(figsize=(6, 4))
        
        # Scatter plot
        for class_index, class_name in enumerate(classes):
            plt.scatter(
                X_df[feature_names[i]][y_numbers == class_index],
                X_df[feature_names[j]][y_numbers == class_index],
                label=class_name,
                edgecolor='k'
            )
        
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        plt.title(f'{feature_names[i]} vs {feature_names[j]}')
        plt.legend()  # Add legend
        plt.show()
