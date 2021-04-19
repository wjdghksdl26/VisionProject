import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef count(np.ndarray[DTYPE_t, ndim=3] p0, np.ndarray[DTYPE_t, ndim=3] p1, np.ndarray[DTYPE_t, ndim=2] err):
    cdef int l = p0.shape[0]
    cdef int i
    cdef int j = 0
    cdef int good_num = 0
    cdef np.ndarray[int, ndim=1] good
    cdef np.ndarray[DTYPE_t, ndim=3] src
    cdef np.ndarray[DTYPE_t, ndim=3] dst
    good = np.zeros(l, dtype=int)
    for i in range(l):
        if 0.0 < err[i][0] < 25.0:
            if p1[i][0][0] < 70.0 or p1[i][0][0] > 354.0:
                good[i] = 1
                good_num += 1
    src = np.zeros((good_num, 1, 2), dtype=p0.dtype)
    dst = np.zeros((good_num, 1, 2), dtype=p0.dtype)
    for i in range(l):
        if good[i] == 1:
            src[j][0] = p0[i][0]
            dst[j][0] = p1[i][0]
            j += 1

    return src, dst
