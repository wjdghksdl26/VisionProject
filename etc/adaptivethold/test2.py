import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

while (1):
    ret, src = cap.read()
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 5, blockSize=3, useHarrisDetector=True, k=0.03)

    for i in corners:
        cv2.circle(src, tuple(i[0]), 3, (0, 0, 255), 2)

    cv2.imshow("dst", src)
    k = cv2.waitKey(0) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()