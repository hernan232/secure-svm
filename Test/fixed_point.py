from fxpmath import Fxp
import numpy as np

x = Fxp(np.array([[1/3, 1/3]]), signed=False, n_word=32, n_frac=20)
y = Fxp(np.array([[1/3], [1/3]]), signed=False, n_word=32, n_frac=20)

print("X =", x.get_val())
print("Y =", y.get_val())

print("Dot prod scaled = ", Fxp(x.get_val().dot(y.get_val()), n_word=32, n_frac=20))