import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target
print("First 10 records:")
print(iris_df.head(10))
print("\nTotal number of rows and columns:")
print(iris_df.shape)
print("\nColumn names:")
print(iris_df.columns.tolist())
print("\nMean of all attributes:")
print(iris_df.mean())
