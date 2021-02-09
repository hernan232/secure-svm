import numpy as np

class FlpDualSVM(object):

    def __init__(self, C, eps=10e-3, kernel="lineal", degree=None) -> None:
        super().__init__()
        self.eps = eps
        self.degree = degree
        self.C = C

    def kernel(self, a, b):
        if self.kernel == "lineal":
            return a.T.dot(b)
        if self.kernel == "poly":
            return np.power(1 + a.T.dot(b), self.degree)
    
    def predict(self, X):
        return np.sign(np.dot(X, self.W) - self.b)

    def predict_dist(self, X):
        return np.dot(X, self.W) - self.b

    def fit(self, X, y):
        self.alphas = np.random.random(size=(X.shape[0], 1))
        self.W = np.random.random(size=(X.shape[0], 1))
        self.b = np.random.rand()

        # Compute error cache
        self.error_cache = self.predict_dist(X) - y

        num_changed = 0
        examine_all = True
        
        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(X.shape[0]):
                    num_changed += self.examine_example(i)
            else:
                non_zero_non_c = self.get_non_zero_non_c_alpha()
                for i in non_zero_non_c:
                    num_changed += self.examine_example(i)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

    def take_step(self, i1, i2):
        pass

    def get_non_zero_non_c_alpha(self):
        indexes = []
        for i in range(self.alphas.shape[0]):
            if self.alphas[i] > 0 and self.alphas[i] < self.C:
                indexes.append(i)
        return indexes

    def get_index_heuristic(self):
        # TODO implement second heuristic
        pass
    
    def examine_example(self, i2, X, y):
        y2 = y[i2][0]
        alph2 = self.alphas[i2][0]
        E2 = self.error_cache[i2][0]
        r2 = E2 * y2

        if (r2 < -self.eps and alph2 < self.C) or (r2 > self.eps and alph2 > 0):
            non_zero_non_c = self.get_non_zero_non_c_alpha()
            if len(non_zero_non_c) > 1:
                i1 = self.get_index_heuristic()
                if self.take_step(i1, i2):
                    return 1
            
            # TODO implement this in a randomized way
            for index in non_zero_non_c:
                if self.take_step(index, i2):
                    return 1
            
            # TODO implement this in a randomized way
            for index in range(X.shape[0]):
                if index != i2 and self.take_step(index, i2):
                    return 1
                
        return 0


    

    