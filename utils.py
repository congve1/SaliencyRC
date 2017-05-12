import numpy as np

def sqr(x):
    return x * x
def sqrDist(l1, l2):
    return np.sum(np.power(np.array(l1,np.float32) - np.array(l2,np.float32),2))
def dist(l1, l2):
    return np.sqrt(sqrDist(l1, l2))