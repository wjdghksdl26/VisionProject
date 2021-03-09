import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef count(np.ndarray[DTYPE_t, ndim=2] tr, int x1, int x2, int y1, int y2, int y3, int y4, int w, int h):
    cdef int reg1=0
    cdef int reg2=0
    cdef int reg3=0
    cdef int reg4=0
    cdef int reg5=0
    cdef int reg6=0
    cdef int i
    cdef int l = tr.shape[0]
    cdef int x, y
    for i in range(l):
        x = tr[i][0]
        y = tr[i][1]

        if 0 < x < x1 and 0 < y < y1:
            reg1 += 1
        if x2 < x < w and 0 < y < y1:
            reg2 += 1
        if 0 < x < x1 and y4 < y < h:
            reg3 += 1
        if x2 < x < w and y4 < y < h:
            reg4 += 1
        if 0 < x < x1 and y2 < y < y3:
            reg5 += 1
        if x2 < x < w and y2 < y < y3:
            reg6 += 1

    return reg1 < 10, reg2 < 10, reg3 < 10 , reg4 < 10, reg5 < 10, reg6 < 10

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef wavg(np.ndarray[DTYPE_t, ndim=2] clusters):
    cdef float rx = 0.0
    cdef float ry = 0.0
    cdef float rw = 0.0
    cdef float rh = 0.0
    cdef int i
    cdef int j
    cdef int nclusters
    cdef int npts
    cdef np.ndarray[DTYPE_t, ndim=2] centerls
    cdef np.ndarray[DTYPE_t, ndim=2] sizels
    nclusters = clusters.shape[0]

    for i in range(nclusters):
        npts = clusters.shape[1]
        rx = 0.0
        ry = 0.0
        rw = 0.0
        rh = 0.0
        
        for j in range(npts):
            rx += clusters[i][j][0] * clusters[i][j][2]
            ry += clusters[i][j][1] * clusters[i][j][3]
            rw += clusters[i][j][2]
            rh += clusters[i][j][3]

        x = rx / rw
        y = ry / rh

        centerls = np.zeros((nclusters, 2), dtype = float)