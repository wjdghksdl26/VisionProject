import numpy as np
import imutils
import cv2
import time

def draw_flow(img, flow, step=30, black=False):
    global width, height

    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if black:
        vis = np.zeros((height, width, 3), np.uint8)

    cv2.polylines(vis, lines, 0, (0, 255, 0))

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 2, (0, 255, 0), -1)

    return vis

def advanced_optflow():
    global width, height

    cap = cv2.VideoCapture('videos/corridor2.mp4')

    ret, prev = cap.read()
    prev = imutils.resize(prev, width=640)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    
    blackscreen = False
    while True:
        ts = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.0, 0.5, 3, 15, 3, 5, 1.1, 0)

        prevgray = gray

        frame2 = draw_flow(gray, flow, black=blackscreen)

        te = time.time()
        FPS = 1 / (te - ts)
        cv2.putText(frame2, str(round(FPS, 1))+"FPS", (20, 65),
                             cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("Grid Optical Flow", frame2)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

advanced_optflow()