from scipy.spatial import distance as dist
from collections import OrderedDict
from collections import deque
import numpy as np


class Tracker():
    def __init__(self, dist_thresh=75, maxDisappeared=5, track_length=10, track_start_length=3):
        self.nextID = 0
        self.tempID = 100
        self.objects = OrderedDict()
        self.objects_TF = OrderedDict()
        self.disappeared = OrderedDict()
        self.dist_thresh = dist_thresh
        self.maxDisappeared = maxDisappeared
        self.track_length = track_length
        self.track_start_length = track_start_length

    def register(self, pt):
        self.objects[self.tempID] = deque([pt], maxlen=self.track_length)
        self.objects_TF[self.tempID] = False
        self.disappeared[self.tempID] = 0
        self.tempID += 1

    def deregister(self, objID):
        del self.objects[objID]
        del self.disappeared[objID]
        if objID not in self.objects_TF:
            return
        if self.objects_TF[objID] == False:
            del self.objects_TF[objID]
            self.tempID -= 1
        else:
            del self.objects_TF[objID]
            truepts = sum(self.objects_TF.values())
            self.nextID = truepts

    def update(self, pts):
        # truepts = sum(self.objects_TF.values())
        # pts = list of tuples
        if len(pts) == 0:
            for i in list(self.disappeared.keys()):
                self.disappeared[i] += 1

                if self.disappeared[i] == self.maxDisappeared:
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

                if D[row][col] < self.dist_thresh:
                    objectID = IDs[row]
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

                    if self.disappeared[objectID] == self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedcols:
                    self.register(pts[col])

            for key in list(self.objects.keys()):
                if len(self.objects[key]) == self.track_start_length:
                    if self.objects_TF[key] == True:
                        continue
                    else:
                        self.objects_TF[self.nextID] = True
                        if key in self.objects_TF:
                            del self.objects_TF[key]
                        self.objects[self.nextID] = self.objects.pop(key)
                        self.disappeared[self.nextID] = self.disappeared.pop(key)
                        truepts = sum(self.objects_TF.values())
                        self.nextID = truepts

        return self.objects
        