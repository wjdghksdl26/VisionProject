import cv2
import numpy as np
from collections import deque
from imgops.optflow_realsense import count


def OpticalFlow(img0, img1, p0, params):
    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **params)

    if p1 is not None:
        # cython boost
        src, dst = count(p0, p1, err)

    if p1 is None:
        src, dst = None, None

    return src, dst