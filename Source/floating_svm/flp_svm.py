import numpy as np
import sklearn


class FlpSVM(object):

    def __init__(self, C, lr=0.01) -> None:
        super().__init__()
        self.C = C
        self.lr = lr

    def fit(self, X, y, epochs=100, verbose=0):
        # Number of features
        m = X.shape[1]
        
        # Append ones at the end of matrix X
        ones = np.ones((X.shape[0], 1)) 
        data = np.append(X, ones, axis=1)

        # Vector of weights
        self.W = np.random.random(size=(m + 1, 1))

        for epoch in range(epochs):
            data, y = sklearn.utils.shuffle(data, y)
            grad = self.compute_loss_grad(data, y)
            self.W = self.W - self.lr * grad

            if verbose == 1:
                print("\t - Epoch:", epoch, " - Cost:", self.loss(data, y))

        return self.W                    

    def predict(self, X):
        ones = np.ones((X.shape[0], 1)) 
        data = np.append(X, ones, axis=1)
        return np.sign(np.dot(data, self.W))

    def loss(self, X, y):
        N = X.shape[0]
        distances = 1 - y * np.dot(X, self.W)
        distances[distances < 0] = 0

        # Compute of Hinge loss
        hinge_loss = self.C * np.sum(distances) / N

        # Calculate cost
        cost = (1 / 2) * np.dot(self.W.T, self.W) + hinge_loss

        return cost.item()

    def compute_loss_grad(self, X, y):
        distance = 1 - y * np.dot(X, self.W)
        dw = np.zeros((len(self.W), 1))

        for index, dist in enumerate(distance):
            if max(0, dist) == 0:
                dist_i = self.W
            else:
                dist_i = self.W - self.C * y[index][0] * np.expand_dims(X[index], axis=1)

            dw += dist_i
        
        dw /= X.shape[0]

        return dw

    def score(self, X, y_true):
        predict = self.predict(X)
        n_correct = np.sum(predict == y_true)
        return n_correct / X.shape[0]







