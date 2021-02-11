from sklearn import datasets
import numpy as np
import flp_svm
import flp_dual_svm
import pandas as pd
import matplotlib.pyplot as plt
import datetime

def generate_dataset(n_samples, n_features):
    X, y = datasets.make_classification(n_samples, n_features, n_redundant=0, n_informative=2)
    y = pd.Series(y).map({0: -1, 1: 1}).values

    # Extend y columns
    y = np.expand_dims(y, axis=1)

    # Save dataset for MATLAB testing
    df_save = pd.DataFrame(data=np.append(X, y, axis=1))
    df_save.to_csv("Source/Datasets/toy_dataset.csv", index=False, columns=None)

    return X, y

def load_dataset(filename):
    df = pd.read_csv("Source/Datasets/" + filename)
    X = df.iloc[:, :2]
    y = df.iloc[:, 2]

    y = np.expand_dims(y, axis=1)

    return X.to_numpy(), y

X, y = generate_dataset(n_features=2, n_samples=500)

# Print shape of dataset
print("X shape =", X.shape)
print("y shape =", y.shape)

svm = flp_svm.FlpSVM(C=4, lr=0.01)
time_a = datetime.datetime.now()
svm.fit(X, y, epochs=20, verbose=1)
print("Fit time linear =", datetime.datetime.now() - time_a)
training_score = svm.score(X, y)
print("Accuracy linear =", training_score)

svm_dual = flp_dual_svm.FlpDualSVM(C=4)
time_a = datetime.datetime.now()
svm_dual.fit(X, y)
print("Fit time dual =", datetime.datetime.now() - time_a)
training_score_dual = svm_dual.score(X, y)
print("Accuracy linear dual =", training_score_dual)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, cmap='viridis')
plt.title("Real dataset")

prediction = svm_dual.predict(X)

plt.figure()
plt.scatter(X[:,0], X[:,1], c=prediction, cmap='viridis')
plt.title("Predictions")

plt.show() 