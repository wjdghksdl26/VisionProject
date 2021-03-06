cdef int count(tr, int x1, int x2, int y1, int y2, int y3, int y4, int w, int h):
    cdef int reg1, reg2, reg3, reg4, reg5, reg6, l
    l = len(tr)
    for i in range(l):
        (x, y) = tr[i]
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

    return reg1, reg2, reg3, reg4, reg5, reg6