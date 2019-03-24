import tensorflow as tf
import numpy as np

def f1():
    a = [[1,1],[1,1],[1,1]]
    b = [[1,1,1],[1,1,1]]
    c = np.dot(b, a)
    print(c)

# f1()

a = 10
b = a
b = 1
print(a)
