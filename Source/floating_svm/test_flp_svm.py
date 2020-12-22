from sklearn import datasets
import numpy as np
import flp_svm
import pandas as pd
import matplotlib.pyplot as plt

X, y = datasets.make_classification(n_samples=500, n_features=2, n_redundant=0, n_informative=2)
y = pd.Series(y).map({0: -1, 1: 1}).values

# Extend y columns
y = np.expand_dims(y, axis=1)

# Save dataset for MATLAB testing
df_save = pd.DataFrame(data=np.append(X, y, axis=1))
print(np.append(X, y, axis=1))
df_save.to_csv("Source/Datasets/toy_dataset.csv", index=False, columns=None)

# Print shape of dataset
print("X shape =", X.shape)
print("y shape =", y.shape)

svm = flp_svm.FlpSVM(C=100, lr=0.01)
svm.fit(X, y, epochs=10, verbose=1)

training_score = svm.score(X, y)
print("Accuracy =", training_score)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
plt.title("Real dataset")

prediction = svm.predict(X)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=prediction, cmap='viridis')
plt.title("Predictions")

plt.show()

