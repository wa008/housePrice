import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

def f1():
    a = [[1,2],[2,3]]
    b = [[4,5],[5,6]]
    d = [1,3,4]
    e = [5,6,4]
    c = np.concatenate((d,e))
    print(c)
    print(type(c))
f1()

