from scipy.spatial import distance as dist
import numpy as np
import time
#list = [(3,8),(7,5),(12,7),(8,5),(12,8),(8,6)]
ts = time.time()
list = [(1.0,1.0),(2.0,1.0),(5.0,2.0),(7.0,2.0),(2.0,3.0),(3.0,6.0),\
    (4.0,6.0),(7.0,6.0),(9.0,6.0),(3.0,7.0),(7.0,7.0),(9.0,7.0),(7.0,9.0),(1.0,10.0),(10.0,10.0),(8.0,7.0)]


m = n = len(list)
c = dict()
thresh = 2.1
k = 1
visited = dict()

for i in range(m):
    c[i] = set()
    if list[i] not in visited:
        c[i].add(list[i])
        visited[list[i]] = i
    for j in range(k, n):
        if dist.euclidean(list[i], list[j]) < thresh:
            if list[j] not in visited:
                c[visited[list[i]]].add(list[j])
                visited[list[j]] = visited[list[i]]
            else:
                c[visited[list[j]]].add(list[i])
                visited[list[i]] = visited[list[j]]
                #del c[i]
                #break
    if len(c[i]) == 0:
        del c[i]

    k = k + 1

print(c, "\n")
for i in c:
    rx = ry = 0
    for (x, y) in c[i]:
        rx += x
        ry += y
    print(c[i])
    print("x", rx/len(c[i]))
    print("y", ry/len(c[i]))
    print("\n")

te = time.time()
print("elapsed time", te-ts)
