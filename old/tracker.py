from scipy.spatial import distance as dist
from collections import OrderedDict
from collections import deque
import numpy as np

class Tracker():
    def __init__(self, thresh=35, maxDisappeared=3, track_length=10):
        self.nextID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.thresh = thresh
        self.maxDisappeared = maxDisappeared
        self.track_length = track_length

    def register(self, pt):
        self.objects[self.nextID] = deque([pt])
        self.disappeared[self.nextID] = 0
        self.nextID += 1

    def deregister(self, objID):
        del self.objects[objID]
        del self.disappeared[objID]
        self.nextID -= 1

    def update(self, pts):
        # pts = list of tuples
        if len(pts) == 0:
            for i in list(self.disappeared.keys()):
                self.disappeared[i] += 1

                if self.disappeared[i] > self.thresh:
                    self.deregister(i)

            return self.objects

        if len(self.objects) == 0:
            for i in pts:
                self.register(i)

        else:
            IDs = list(self.objects.keys())
            cents = list(self.objects.values())
            calc = [c[-1] for c in cents]

            D = dist.cdist(calc, pts)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedrows = set()
            usedcols = set()

            for (row, col) in zip(rows, cols):
                if row in usedrows or col in usedcols:
                    continue

                if D[row][col] < self.thresh:
                    objectID = IDs[row]
                    if len(self.objects[objectID]) == self.track_length:
                        self.objects[objectID].popleft()
                        self.objects[objectID].append(pts[col])
                        self.disappeared[objectID] = 0
                    else:
                        self.objects[objectID].append(pts[col])
                        self.disappeared[objectID] = 0
                    usedrows.add(row)
                    usedcols.add(col)

            unusedrows = set(range(0, D.shape[0])).difference(usedrows)
            unusedcols = set(range(0, D.shape[1])).difference(usedcols)

            if D.shape[0] > D.shape[1]:
                for row in unusedrows:
                    objectID = IDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedcols:
                    self.register(pts[col])

        return self.objects
