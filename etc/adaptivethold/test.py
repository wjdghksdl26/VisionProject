import cv2
import argparse
import numpy as np
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

i = 0
ret1, frame1 = cap.read()

while (1):
    orig = frame1.copy()
    ret2, frame2 = cap.read()
    f2 = frame2.copy()
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.adaptiveThreshold(frame1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 31, 65)
    frame2 = cv2.adaptiveThreshold(frame2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                   cv2.THRESH_BINARY, 31, 65)

    th2 = frame1-frame2
    kernel1 = np.ones((3, 3), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    #th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel1)
    #th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel2)

    th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
    #final = np.hstack((orig, th2))
    #final = imutils.resize(final, width = 1400)
    cv2.imshow('frame', th2)
    cv2.imshow('frame2', orig)
    #cv2.imshow("result", final)
    print(i)
    i+=1
    frame1 = f2
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()