import cv2
import imutils
import numpy as np
import argparse
import time

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict(maxCorners=30, qualityLevel=0.3, minDistance=50, blockSize=5, useHarrisDetector=False)
lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=termination, minEigThreshold=1e-5)
search_new_points = False
switch_regions = False

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())

if args["video"] == "cam":
    video = 0
else:
    video = args["video"]


class App:
    def __init__(self):
        self.vid = cv2.VideoCapture(video)
        self.track_len = 20
        self.detect_interval = 5
        self.tracks = []
        self.frame_idx = 1
        self.totalTime = 0
        self.find_new_pts = search_new_points
        self.new_pts_switch = True
        self.track_fullscreen = False
        self.discard_old_regions = switch_regions
        self.initBB = None

    def run(self):
        while True:
            t_start = time.time()
            ret, frame = self.vid.read()
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            if not ret:
                print("End of Video Stream!")
                break

            frame = imutils.resize(frame, width=400)
            blurred = cv2.GaussianBlur(frame, (9, 9), 0)
            print("Frame", self.frame_idx)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []

                for tr, (x,y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue

                    tr.append((x,y))

                    if len(tr) > self.track_len:
                        del tr[0]

                    new_tracks.append(tr)
                    cv2.circle(blurred, (int(x), int(y)), 4, (0, 0, 255), -1)

                self.tracks = new_tracks
                cv2.polylines(blurred, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))


            if self.frame_idx % self.detect_interval == 0 and self.new_pts_switch == True:
                if self.track_fullscreen is False:
                    if self.initBB is None:
                        self.tracks = []

                    mask = np.zeros_like(frame_gray)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

                if self.track_fullscreen is True:
                    mask = np.ones_like(frame_gray)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

                if self.initBB is not None:
                    if self.discard_old_regions is True:
                        self.tracks = []
                    mask = np.zeros_like(frame_gray)
                    mask[int(bby):int(bby+bbh), int(bbx):int(bbx+bbw)] = 1
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

                    if self.find_new_pts == False:
                        self.new_pts_switch = False

                if p is not None:
                    for x, y in p.reshape(-1, 2):
                        self.tracks.append([(x,y)])


            self.frame_idx += 1
            self.prev_gray = frame_gray

            cv2.imshow('frame', blurred)
            k = cv2.waitKey(1) & 0xFF

            if k == ord("s"):
                self.initBB = cv2.selectROI('frame', vis, fromCenter=False, showCrosshair=True)
                (bbx, bby, bbw, bbh) = self.initBB
                self.new_pts_switch = True

            if k == ord("f"):
               self.track_fullscreen =  not self.track_fullscreen

            if k == 27:
                break

            t_end = time.time()
            print("Elapsed time:", t_end - t_start, "\n")
            self.totalTime = self.totalTime + (t_end - t_start)

        print("Avg. processing time per frame =", self.totalTime / self.frame_idx)
        self.vid.release()


video_src = 1
App().run()
cv2.destroyAllWindows()
