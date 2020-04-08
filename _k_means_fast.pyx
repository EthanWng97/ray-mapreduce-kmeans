import numpy as np
cimport numpy as np
cimport cython
from cpython cimport array as cparray

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[np.float32_t, ndim=2] _naive_dot(np.ndarray[np.float32_t, ndim=2] a, np.ndarray[np.float32_t, ndim=2] b):
    cdef np.ndarray[np.float32_t, ndim=2] c
    cdef int n, p, m
    cdef np.float32_t s
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    n, p, m = a.shape[0], a.shape[1], b.shape[1]
    c = np.zeros((n, m), dtype=np.float32)
    for i in xrange(n):
        for j in xrange(m):
            s = 0
            for k in xrange(p):
                s += a[i, k] * b[k, j]
            c[i, j] = s
    return c

def naive_dot(a, b):
    return _naive_dot(a, b)

cdef np.float32_t _calEDist(np.ndarray[np.float32_t, ndim=1] a, np.ndarray[np.float32_t, ndim=1] b):
    return np.sqrt(sum(np.power(a-b, 2)))

def calEDist(a,b):
    return _calEDist(a,b)


cdef _createDistMatrix(double[:,:] center, double[:,:] newcenter):
    cdef int n = center.shape[0]
    for i in range(n-1):
        newcenter[i][i+1] = np.sqrt(np.square(center[i][0]-center[i+1][0])+np.square(center[i][1]-center[i+1][1]))

def createDistMatrix(center,newcenter):
    return _createDistMatrix(center,newcenter)