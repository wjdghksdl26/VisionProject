from sklearn.cluster import OPTICS
import numpy as np
import time

t_start = time.time()
for i in range(3):
    #X = np.array([[1,2],[2,5],[3,6],[2,4],[8,7],[8,8],[7,3]])
    X = np.array([[1,2],[100,7]])
    cluster = OPTICS(min_samples=2, cluster_method='dbscan', eps=5).fit(X)
    print(cluster.labels_)
t_end = time.time()

print((t_end-t_start)/3)