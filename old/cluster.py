import numpy as np
from scipy.spatial import distance as dist
from logicops.count import wavg


def cluster(pts, thresh=30.0):
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


def clusterWithSize(pts, thresh=30.0):
    n = len(pts)
    clusters = dict()
    visited = dict()
    k = 1

    for i, pt in enumerate(pts):
        clusters[i] = set()
        if pt not in visited:
            clusters[i].add(pt)
            visited[pt] = i
        
        for j in range(k, n):
            if dist.euclidean(pt[0], pts[j][0]) < thresh:
                if pts[j] not in visited:
                    clusters[visited[pt]].add(pts[j])
                    visited[pts[j]] = visited[pt]
                else:
                    clusters[visited[pts[j]]].add(pt)
                    visited[pt] = visited[pts[j]]

        if len(clusters[i]) == 0:
            del clusters[i]

        k = k + 1
    
    centerls = []
    sizels = []
    
    cl = []
    for idx, i in enumerate(clusters):
        cl.append([np.asarray([j[0][0], j[0][1], j[1], j[2]]) for j in clusters[i]])
        cl[idx] = np.vstack(cl[idx])

    # cython boost
    if len(cl) != 0:
        for c in cl:
            xy, wh = wavg(c)
            centerls.append(xy)
            sizels.append(wh)

    return centerls, sizels