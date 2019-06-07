import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
def f1():
    a = [[1,2],[2,3]]
    b = [[4,5],[5,6]]
    d = [1,3,4]
    e = [5,6,4]
    c = np.concatenate((d,e))
    print(c)
    print(type(c))
# f1()

def f2():
    a = list(i*0.01 for i in range(1,32,5))
    print(a)

# f2()

def f3():
    t = []
    t.append(time.clock())
    time.sleep(1)
    t.append(time.clock())
    time.sleep(1)
    t.append(time.clock())
    time.sleep(1)
    t.append(time.clock())
    time.sleep(1)
    t.append(time.clock())
    time.sleep(1)
    t.append(time.clock())
    time.sleep(1)
    t.append(time.clock())
    time.sleep(1)
    for i in range(len(t)-1):
        print(t[i+1]-t[i])

# f3()

def f4():
    x = np.array([1,2,3])
    y = np.array([3,2,1])
    print(10 - x)
    print(max(x,y))

if __name__ == '__main__':
    f4()