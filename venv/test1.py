import tensorflow as tf
import numpy as np

def f1():
    a = [[1,1],[1,1],[1,1]]
    b = [[1,1,1],[1,1,1]]
    c = np.dot(b, a)
    print(c)

# f1()

a = np.ones((2,3))
b = np.ones((3,2))
d = 1
c = a.dot(b) + d

# print(a)

