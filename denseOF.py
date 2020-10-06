import numpy as np
import argparse
import cv2
import imutils

feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=7, blockSize=7, useHarrisDetector=False)
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=termination, minEigThreshold=1e-4)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())

def denseOF():
    cap = cv2.VideoCapture(args["video"])

    ret, frame = cap.read()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = imutils.resize(frame, width = 200)
    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame)
    hsv[...,1] = 255

    while True:
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = imutils.resize(frame, width=200)
        #frame = cv2.GaussianBlur(frame, (9, 9), 0)
        next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, next, 0.0, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame', rgb)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        prev = next

    cap.release()
    cv2.destroyAllWindows()

denseOF()