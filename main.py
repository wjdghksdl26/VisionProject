import numpy as np
import cv2
import imutils
import argparse
import time
from matplotlib import pyplot as plt
from imgops.subtract_imgs import subtract_images
from imgops.get_optflow_mod import opticalflow

# file name and location
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())

# parameters for feature point detection
feature_params = dict(maxCorners=30, qualityLevel=0.1, minDistance=7, blockSize=7, useHarrisDetector=False)

# parameters for optical flow
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
lk_params = dict(winSize=(9, 9), maxLevel=3, criteria=termination, minEigThreshold=1e-4)

# parameters for drawing optical flow
track_len = 10
detect_interval = 3


class BGSubt:
    def __init__(self, videopath=1):
        self.vid = cv2.VideoCapture(videopath)
        self.frame_idx = 0
        self.totalTime = 0
        self.tracks = []
        self.kernel = np.ones((2, 2), np.uint8)

    def bgsubtraction(self):

        HMatQidx = [[], []]
        samples = [np.array([[30], [30], [1]]), np.array([[30], [60], [1]]),
                   np.array([[60], [30], [1]]), np.array([[60], [60], [1]])]
        noiselist = []
        fplist = []

        while True:
            t_start = time.time()
            print("Frame", self.frame_idx)
            ret, frame = self.vid.read()

            if not ret:
                print("End of video stream!")
                break

            color = frame.copy()
            color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
            color = imutils.resize(color, width=300)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_gray = cv2.equalizeHist(frame_gray)
            frame_gray = cv2.rotate(frame_gray, cv2.ROTATE_90_CLOCKWISE)
            frame_gray = imutils.resize(frame_gray, width=300)

            if self.frame_idx != 0:
                frame_subt_gray = subtract_images(prev_gray, frame_gray, absolute=True, clip=0)

                if self.frame_idx != 1:
                    new_tracks = opticalflow(prev_subt_gray, frame_subt_gray, self.tracks, lk_params, track_length=track_len)

                    if len(new_tracks) < 4:
                        vis = subtract_images(prev_gray, frame_gray)
                        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                        print("Motion compensation failure!")

                    else:
                        self.tracks = new_tracks
                        src = np.float32([[list(tr[-2])] for tr in self.tracks])
                        dst = np.float32([[list(tr[-1])] for tr in self.tracks])

                        hmat, status = cv2.findHomography(src, dst, cv2.RANSAC, 7.5)
                        h, w = frame_gray.shape
                        warped = cv2.warpPerspective(prev_gray, hmat, (w, h))
                        vis = subtract_images(warped, frame_gray, absolute=True, clip=0)

                        _, thold = cv2.threshold(vis, 50, 255, cv2.THRESH_BINARY)

                        total = 0
                        for s in samples:
                            total += np.linalg.norm(np.dot(hmat, s) - s)

                        HMatQidx[0].append(self.frame_idx)
                        HMatQidx[1].append(np.sqrt(total))

                        noisecalc = vis[20:h - 20, 20:w - 20]
                        noise = np.sum(noisecalc - 50)
                        noiselist.append(noise)
                        fplist.append(len(self.tracks))

                        thold = cv2.cvtColor(thold, cv2.COLOR_GRAY2BGR)
                        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                        cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))


                prev_subt_gray = frame_subt_gray

            prev_gray = frame_gray

            if self.frame_idx % detect_interval == 1:
                mask = np.ones_like(frame_gray)
                #mask = np.zeros_like(frame_gray)
                #mask[:, 0:100] = 1
                #mask[:, -100:0] = 1
                p = cv2.goodFeaturesToTrack(prev_subt_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in p.reshape(-1, 2):
                        self.tracks.append([(x, y)])

            if self.frame_idx > 2:

                # vis = cv2.adaptiveThreshold(vis, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 10)

                '''
                vis_inv = cv2.bitwise_not(vis)
                params = cv2.SimpleBlobDetector_Params()
                params.minThreshold = 0
                params.maxThreshold = 255
                params.filterByArea = True
                params.minArea = 40
                params.filterByInertia = True
                params.minInertiaRatio = 0.02
                params.filterByColor = False
                params.filterByCircularity = False
                params.filterByConvexity = True
                params.minConvexity = 0.5
                detector = cv2.SimpleBlobDetector_create(params)
                kpts = detector.detect(vis_inv)
                vis = cv2.drawKeypoints(vis, kpts, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                '''


                '''
                # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(vis)
                '''

                '''
                contours, hierarchy = cv2.findContours(vis, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 10:
                        cv2.drawContours(vis, [cnt], 0, (255, 0, 0), 2)
                '''

                final = np.hstack((color, vis))
                # final = cv2.rotate(final, cv2.ROTATE_90_COUNTERCLOCKWISE)

                cv2.imshow('result',final)

            self.frame_idx += 1
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                print("User interrupt!")
                break

            t_end = time.time()
            print("Elapsed time:", "%.3f" % round(t_end - t_start, 3), "sec\n")
            self.totalTime += (t_end - t_start)

        print("Avg. processing time per frame =", round(self.totalTime / self.frame_idx, 3), "sec\n")

        self.vid.release()

        plt.subplot(131)
        plt.plot(HMatQidx[0], HMatQidx[1])
        plt.xlabel('Frame')
        plt.title('Homography Matrix')
        plt.ylim(0, 100)

        plt.subplot(132)
        plt.plot(HMatQidx[0], noiselist)
        plt.xlabel('Frame')
        plt.title('Total Noise - Lower is better')
        # plt.ylim(0, 100000)

        plt.subplot(133)
        plt.plot(HMatQidx[0], fplist)
        plt.xlabel('Frame')
        plt.title('No. of Feature Points')
        plt.ylim(0, 100)

        plt.show()

        return HMatQidx[0], HMatQidx[1], noiselist, fplist


App = BGSubt(args["video"])
#App = BGSubt()
App.bgsubtraction()
cv2.destroyAllWindows()
