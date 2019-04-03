import tensorflow as tf
import numpy as np

def f1():
    a = np.array([5,2,5,2])
    b = np.array([1,3,4,1])
    print(abs(a-b)/b)
    print((abs(a-b)/b).sum())

f1()
