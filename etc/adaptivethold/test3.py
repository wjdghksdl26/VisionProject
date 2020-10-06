import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())

cap = cv2.VideoCapture(args["video"])

i = 0
while (1):
    ret1, frame1 = cap.read()
    orig = frame1.copy()

    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    frame1 = cv2.adaptiveThreshold(frame1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY, 91, 75)

    th2 = frame1
    kernel1 = np.ones((5, 5), np.uint8)
    kernel2 = np.ones((5, 5), np.uint8)
    #th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel1)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel2)
    cv2.imshow('frame', cv2.bitwise_not(th2))
    cv2.imshow('frame2', orig)
    print(i)
    i+=1
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()