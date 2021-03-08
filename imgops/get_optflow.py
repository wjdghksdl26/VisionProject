import cv2
import numpy as np
from collections import deque


def OpticalFlow(img0, img1, tracks, params):
    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **params)
    #err = [i[0] for i in err]

    if p1 is not None:
        err = err.reshape(-1, 1)
        #p0r, st, err_ = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **params)
        #d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        #print(d)
        #print(err)
        #good = d < 0.7
        #print(good)
        good_ = ((err > 0) & (err < 25))
        #print(good_)
        #print("Match :", good & good_)
        new_tracks = deque()
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good_):
            if not good_flag:
                continue

            if 70.0 < x < 250.0 or x < 3.0 or x > 317.0:
                continue

            if 70.0 < y < 85.0 or 155.0 < y < 170.0 or y < 3.0 or y > 237.0:
                continue

            tr.append((x, y))

            new_tracks.append(tr)

    if p1 is None:
        new_tracks = deque()

    return new_tracks