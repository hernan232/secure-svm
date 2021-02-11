import numpy as np

class FlpDualSVM(object):

    def __init__(self, C, eps=1e-3, kernel="linear", tolerance=1e-3, degree=None) -> None:
        super().__init__()
        self.eps = eps
        self.degree = degree
        self.C = C
        self.tolerance = tolerance
        self.kernel_type = kernel

    def kernel(self, a, b):
        if self.kernel_type == "linear":
            return a.T.dot(b)[0]
        if self.kernel_type == "poly":
            return np.power(1 + a.T.dot(b)[0], self.degree)
    
    def predict_distance_vect(self, a):
        return np.dot(self.W.T, a) - self.b

    def predict_distance(self, X):
        return np.dot(X, self.W) - self.b

    def predict(self, X):
        distances = self.predict_distance(X)
        return np.sign(distances)

    def update_weights(self):
        self.W = np.dot(self.data.T, np.multiply(self.alphas, self.y))

    def fit(self, X, y):
        self.data = X
        self.y = y
        self.steps = 0
        self.alphas = np.random.random(size=(X.shape[0], 1))
        self.update_weights()

        for i in range(self.alphas.shape[0]):
            if self.alphas[i] > 0:
                Xi = np.expand_dims(self.data[i], axis=1)
                self.b = np.dot(self.W.T, Xi) - self.y[i]
                break

        num_changed = 0
        examine_all = True

        while num_changed > 0 or examine_all:
            num_changed = 0
            if examine_all:
                for i in range(self.data.shape[0]):
                    num_changed += self.examine_example(i)
            else:
                non_zero_non_c = self.get_non_zero_non_c_alpha()
                for i in non_zero_non_c:
                    num_changed += self.examine_example(i)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

        print("Steps =", self.steps)

        if self.kernel_type != "linear":
            self.update_weights()

    def take_step(self, i1, i2):
        self.steps += 1
        if i1 == i2:
            return 0

        # i2 info
        alph1 = self.alphas[i1][0]
        y1 = self.y[i1][0]
        X1 = np.expand_dims(self.data[i1], axis=1)
        E1 = self.predict_distance_vect(X1) - y1

        # i1 info
        y2 = self.y[i2][0]
        alph2 = self.alphas[i2][0]
        X2 = np.expand_dims(self.data[i2], axis=1)
        E2 = self.predict_distance_vect(X2) - y2

        s = y1 * y2

        # Computing L and H
        if y1 != y2:
            L = max(0, alph2 - alph1)
            H = min(self.C, self.C + alph2 - alph1)
        else:
            L = max(0, alph2 + alph1 - self.C)
            H = min(self.C, alph2 + alph1)

        if L == H:
            return 0

        k11 = self.kernel(X1, X1)
        k12 = self.kernel(X1, X2)
        k22 = self.kernel(X2, X2)

        eta = k11 + k22 - 2 * k12

        if eta > 0:
            # Alpha2 new
            a2 = alph2 + y2 * (E1 - E2) / eta

            # Alpha2 new clipped
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # TODO verify if necessary
            f1 = y1 * (E1 + self.b) - alph1 * k11 - s * alph2 * k12
            f2 = y2 * (E2 + self.b) - s * alph1 * k12 - alph2 * k22
            L1 = alph1 + s * (alph2 - L)
            H1 = alph1 + s * (alph2 - H)

            L_obj = L1 * f1 + L * f2 + (1. / 2.) * (L1 ** 2) * k11 + (1. / 2.) * (L ** 2) * k22 + s * L * L1 * k12
            H_obj = H1 * f1 + H * f2 + (1. / 2.) * (H1 ** 2) * k11 + (1. / 2.) * (H ** 2) * k22 + s * H * H1 * k12

            if L_obj < L_obj - self.eps:
                a2 = L
            elif L_obj > H_obj + self.eps:
                a2 = H
            else:
                a2 = alph2

        if np.abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps):
            return 0
        
        a1 = alph1 + s * (alph2 - a2)

        # Update threshold
        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self.b

        if b1 == b2:
            self.b = b1
        elif (a1 == self.C or a1 == 0) and (a2 == self.C or a2 == 0):
            self.b = (b1 + b2) / 2.
        
        # Update lagrange multipliers vector
        self.alphas[i1][0] = a1
        self.alphas[i2][0] = a2

        # Update weigth vector
        if self.kernel_type == "linear":
            self.W = self.W + y1 * (a1 - alph1) * X1 + y2 * (a2 - alph2) * X2
        else:
            self.update_weights()

        return 0

    def get_non_zero_non_c_alpha(self):
        return np.where(self.alphas > 0)[0]

    def get_index_heuristic(self, i2):
        non_zero_non_c_indexes = self.get_non_zero_non_c_alpha()

        X2 = np.expand_dims(self.data[i2], axis=1)
        E2 = self.predict_distance_vect(X2) - self.y[i2]

        X0 = np.expand_dims(self.data[non_zero_non_c_indexes[0]], axis=1)
        E0 = self.predict_distance_vect(X0) - self.y[non_zero_non_c_indexes[0]][0]

        max_index = 0
        max_error = np.abs(E2 - E0)
        for index in non_zero_non_c_indexes[1:]:
            Xi = np.expand_dims(self.data[index], axis=1)
            Ei = self.predict_distance_vect(Xi) - self.y[index]
            error = np.abs(E2 - Ei)
            if error > max_error and error > 0:
                max_error = error
                max_index = index
       
        if max_error > 0:
            return max_index
        
        X0 = np.expand_dims(self.data[0], axis=1)   
        E0 = self.predict_distance_vect(X0) - self.y[0][0]

        max_index = 0
        max_error = np.abs(E2 - E0)
        for index in range(1, self.data.shape[0]):
            Xi = np.expand_dims(self.data[index], axis=1)
            Ei = self.predict_distance_vect(Xi) - self.y[index]
            error = np.abs(E2 - Ei)
            if error > max_error and error > 0:
                max_error = error
                max_index = index

        if max_error > 0:
            return max_index
            
        return -1
    
    def examine_example(self, i2):
        y2 = self.y[i2][0]
        alph2 = self.alphas[i2][0]
        X2 = np.expand_dims(self.data[i2], axis=1)
        E2 = self.predict_distance_vect(X2) - y2

        r2 = E2 * y2

        if (r2 < -self.tolerance and alph2 < self.C) or (r2 > self.tolerance and alph2 > 0):
            non_zero_non_c = self.get_non_zero_non_c_alpha()
            if len(non_zero_non_c) > 1:
                i1 = self.get_index_heuristic(i2)
                if i1 >= 0 and self.take_step(i1, i2):
                    return 1
                elif i1 < 0:    
                    return 0
            
            # This implements a loop over non-zero and non-C alphas starting at a random point
            rnd_start = np.random.randint(0, len(non_zero_non_c))
            for i in range(len(non_zero_non_c)):
                rnd_i = (i + rnd_start) % len(non_zero_non_c)
                index = non_zero_non_c[rnd_i]
                if self.take_step(index, i2):
                    return 1

            # This implements a loop over all training examples starting at a random point
            rnd_start = np.random.randint(0, self.data.shape[0])
            for index in range(self.data.shape[0]):
                rnd_index = (index + rnd_start) % self.data.shape[0]
                if rnd_index != i2 and self.take_step(rnd_index, i2):
                    return 1
                
        return 0

    def score(self, X, y_true):
        predict = self.predict(X)
        n_correct = np.sum(predict == y_true)
        return n_correct / X.shape[0]


    

    