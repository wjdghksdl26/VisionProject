from sklearn.cluster import OPTICS
import numpy as np
import time

t_start = time.time()
for i in range(3):
    #X = np.array([[1,2],[2,5],[3,6],[2,4],[8,7],[8,8],[7,3]])
    X = np.array([(1.0,1.0),(2.0,1.0),(5.0,2.0),(7.0,2.0),(2.0,3.0),(3.0,6.0),\
    (4.0,6.0),(7.0,6.0),(9.0,6.0),(3.0,7.0),(7.0,7.0),(9.0,7.0),(7.0,9.0),(1.0,10.0),(10.0,10.0),(8.0,7.0)])
    cluster = OPTICS(min_samples=2, cluster_method='dbscan', eps=5).fit(X)
    print(cluster.labels_)
t_end = time.time()

print((t_end-t_start)/3)