import numpy as np
cimport numpy as np
cimport cython

ctypedef np.int_t DTYPE_t
ctypedef np.float_t FTYPE_t

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
cpdef wavg(np.ndarray[FTYPE_t, ndim=2] cluster):
    cdef float rx = 0.0
    cdef float ry = 0.0
    cdef float rw = 0.0
    cdef float rh = 0.0
    cdef int i
    cdef int npts
    npts = cluster.shape[0]

    for i in range(npts):
        
        rx += cluster[i][0] * cluster[i][2]
        ry += cluster[i][1] * cluster[i][3]
        rw += cluster[i][2]
        rh += cluster[i][3]

    x = rx / rw
    y = ry / rh

    return (x, y), (rw, rh)