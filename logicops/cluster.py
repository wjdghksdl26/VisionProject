import numpy as np
from scipy.spatial import distance as dist


def cluster(pts, thresh=3):
    m = n = len(pts)
    cluster = dict()
    visited = dict()
    k = 1

    for i in range(m):
        cluster[i] = set()
        if pts[i] not in visited:
            cluster[i].add(pts[i])
            visited[pts[i]] = i

        for j in range(k, n):
            if dist.euclidean(pts[i], pts[j]) < thresh:
                if pts[j] not in visited:
                    cluster[visited[pts[i]]].add(pts[j])
                    visited[pts[j]] = visited[pts[i]]
                else:
                    cluster[visited[pts[j]]].add(pts[i])
                    visited[pts[i]] = visited[pts[j]]
        
        if len(cluster[i]) == 0:
            del cluster[i]

        k = k + 1

    ls = [np.mean(np.array(list(cluster[i])), axis=0) for i in cluster]

    return ls


def clusterwithsize(pts, thresh=3):
    m = n = len(pts)
    cluster = dict()
    visited = dict()
    k = 1

    for i in range(m):
        cluster[i] = set()
        if pts[i] not in visited:
            cluster[i].add(pts[i])
            visited[pts[i]] = i

        for j in range(k, n):
            if dist.euclidean(pts[i][0], pts[j][0]) < thresh:
                if pts[j] not in visited:
                    cluster[visited[pts[i]]].add(pts[j])
                    visited[pts[j]] = visited[pts[i]]
                else:
                    cluster[visited[pts[j]]].add(pts[i])
                    visited[pts[i]] = visited[pts[j]]
        
        if len(cluster[i]) == 0:
            del cluster[i]

        k = k + 1
    
    centerls = []
    sizels = []
    for i in cluster:
        rx = 0.0
        ry = 0.0
        rs = 0.0
        group = list(cluster[i])

        #weighed average
        for j in group:
            #rx = rx + j[0][0]
            #ry = ry + j[0][1]
            rx = rx + j[0][0] * j[1]
            ry = ry + j[0][1] * j[1]
            rs = rs + j[1]

        #x = rx / len(group)
        #y = ry / len(group)
        x = rx / rs
        y = ry / rs
        centerls.append((int(x), int(y)))
        sizels.append(int(rs))

    # ls = [np.mean(np.array(list(cluster[i])), axis=0) for i in cluster]

    return centerls, sizels