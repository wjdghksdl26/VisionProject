import cv2
import numpy as np


def opticalflow(img0, img1, tracks, params, track_length=10):
    p0 = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **params)

    if p1 is not None:
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 0.7
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue

            if 100 < x < 200:
                continue

            tr.append((x, y))
            if len(tr) > track_length:
                del tr[0]

            new_tracks.append(tr)

    if p1 is None:
        new_tracks = []

    return new_tracks
