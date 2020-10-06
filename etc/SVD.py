import cv2
import argparse
import numpy as np
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

k = 3

while (1):
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = imutils.resize(frame, width = 800)
    U, S, V = np.linalg.svd(frame)
    rcst = np.dot(U[:, :k], np.dot(np.diag(S[:k]), V[:k, :]))
    rcst = rcst.astype(np.uint8)
    print(rcst)
    cv2.imshow('frame', rcst)
    k = cv2.waitKey(0) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()