import numpy as np
from scipy.spatial import distance as dist


def cluster(pts, thresh=30):
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
    for cluster in clusters:
        rx = 0.0
        ry = 0.0
        rw = 0.0
        rh = 0.0
        group = list(clusters[cluster])

        #weighed average
        for pt in group:
            rx = rx + pt[0][0] * pt[1]
            ry = ry + pt[0][1] * pt[2]
            rw = rw + pt[1]
            rh = rh + pt[2]

        x = rx / rw
        y = ry / rh
        #centerls.append((int(x), int(y)))
        #sizels.append((int(rw), int(rh)))
        centerls.append((x, y))
        sizels.append((rw, rh))

    return centerls, sizels