import cv2
import imutils
import numpy as np
import argparse
import time
from collections import deque
from imgops.subtract_imgs import SubtractImages
from imgops.get_optflow import OpticalFlow
from imgops.videostream import VideoStream
from logicops.cluster import clusterWithSize
from logicops.tracker import Tracker
from logicops.kalman2 import Kfilter

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict(maxCorners=15, qualityLevel=0.01, minDistance=3, blockSize=7, useHarrisDetector=False)
lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=termination, minEigThreshold=1e-4)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())

if args["video"] == "cam":
    video = VideoStream(src=0).start()
else:
    video = cv2.VideoCapture(args["video"])

np.set_printoptions(precision=3, suppress=True)


class App:
    def __init__(self, videoPath):
        self.track_len = 5
        self.detect_interval = 1
        self.mask_size = 70
        self.tracks = deque()
        self.vid = videoPath
        self.frame_idx = 0
        self.initiate_kalmanFilter = 12

    def run(self):
        # images for initialization
        ret, frame1 = self.vid.read()
        frame1 = imutils.resize(frame1, width=320)
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

        ret, frame2 = self.vid.read()
        frame2 = imutils.resize(frame2, width=320)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # video size
        h, w = frame1.shape

        # masks for feature point search
        x1 = self.mask_size
        x2 = w - self.mask_size
        y1 = self.mask_size
        y2 = int(h/2 - self.mask_size/2)
        y3 = int(h/2 + self.mask_size/2)
        y4 = h - self.mask_size
        # top left
        mask1 = np.zeros_like(frame1)
        mask1[0:y1, 0:x1] = 1
        # top right
        mask2 = np.zeros_like(frame1)
        mask2[0:y1, x2:w] = 1
        # bottom left
        mask3 = np.zeros_like(frame1)
        mask3[y4:h, 0:x1] = 1
        # bottom right
        mask4 = np.zeros_like(frame1)
        mask4[y4:h, x2:w] = 1
        # middle left
        mask5 = np.zeros_like(frame1)
        mask5[y2:y3, 0:x1] = 1
        # middle right
        mask6 = np.zeros_like(frame1)
        mask6[y2:y3, x2:w] = 1

        # performance index variables
        totalFPS = 0

        # kernel for morphology operations
        kernel = np.ones((3, 3))
        #kernel = np.ones((5, 5))
        #gaussiankernel = (3, 3)

        # object tracker initialization
        tracker = Tracker()

        # main loop
        while True:
            t_start = time.time()

            # read and process frame
            ret, frame3 = self.vid.read()
            frame3 = imutils.resize(frame3, width=320)
            if not ret:
                print("End of video stream!")
                break

            # current frame
            vis = frame3.copy()
            frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
            

            # copy of current frame (for visualization)
            #vis = imutils.resize(vis, width=300)
            vis = vis[15:h-15, 15:w-15]

            # begin motion estimation
            if self.frame_idx > 0:
                img1, img2, img3 = frame1, frame2, frame3

                # optical flow from img2 to img3
                new_tracks = OpticalFlow(img2, img3, self.tracks, lk_params)

                # update track
                self.tracks = new_tracks

                # points in img3
                src23 = np.float32([[list(tr[-1])] for tr in self.tracks])
                # points in img2
                dst23 = np.float32([[list(tr[-2])] for tr in self.tracks])

                if len(dst23) >= 12:
                    # Homography Mat. that warps img1 to fit img2
                    HMat1to2 = HMat3to2
                    # Homography Mat. that warps img3 to fit img2
                    HMat3to2, stat = cv2.findHomography(src23, dst23, cv2.RANSAC, 1.0)

                    # current frame
                    print("Frame", self.frame_idx)

                    # warping operation
                    HMat1to2 = np.linalg.inv(HMat1to2)
                    warped1to2 = cv2.warpPerspective(img1, HMat1to2, (w, h), cv2.INTER_LINEAR)
                    # OpenCV 3.x does not have cv2.WARP_INVERSE_MAP
                    #warped1to2 = cv2.warpPerspective(img1, HMat1to2, (w, h), cv2.INTER_LINEAR, cv2.WARP_INVERSE_MAP)
                    warped3to2 = cv2.warpPerspective(img3, HMat3to2, (w, h), cv2.INTER_LINEAR)

                    # Gaussian blur operation to ease impact of edges
                    # warped1to2 = cv2.GaussianBlur(warped1to2, gaussiankernel, 0)
                    # warped3to2 = cv2.GaussianBlur(warped3to2, gaussiankernel, 0)
                    # img2 = cv2.GaussianBlur(img2, gaussiankernel, 0)

                    # subtracted images
                    subt21 = SubtractImages(warped1to2, img2, clip=15)
                    subt23 = SubtractImages(warped3to2, img2, clip=15)

                    # merge subtracted images
                    subt21 = subt21[15:h - 15, 15:w - 15]
                    subt23 = subt23[15:h - 15, 15:w - 15]
                    subt21 = cv2.medianBlur(subt21, 3)
                    subt23 = cv2.medianBlur(subt23, 3)
                    # subt21 = cv2.erode(subt21, kernel)
                    # subt23 = cv2.erode(subt23, kernel)
                    # subt21 = cv2.dilate(subt21, kernel, iterations=1).astype('int32')
                    # subt23 = cv2.dilate(subt23, kernel, iterations=1).astype('int32')
                    merged = (subt21 + subt23) / 2
                    merged = np.where(merged <= 40, 0, merged).astype('uint8')
                    # merged = merged * 2
                    # merged = cv2.dilate(merged, kernel, iterations=1)

                    # ---------- essential operations finished ----------

                    # crude thresholding type 1
                    thold1 = merged.copy()
                    thold1 = cv2.erode(thold1, kernel, iterations=1)
                    _, thold1 = cv2.threshold(thold1, 40, 255, cv2.THRESH_BINARY)
                    thold1 = cv2.dilate(thold1, kernel, iterations=2)

                    # draw flow
                    for tr in self.tracks:
                            cv2.circle(vis, tuple(np.int32(tr[-1])), 2, (0, 0, 255), -1)
                    cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

                # in case of motion compensation failure
                if len(dst23) < 12:
                    print("Motion Compensation Failure!")
                    thold1 = np.zeros_like(img1)
                    thold1 = thold1[15:h-15, 15:w-15]
                    thold1 = thold1.astype('uint8')
                    merged = thold1

            # search feature points
            if self.frame_idx % self.detect_interval == 0:
                # after initialization
                if self.frame_idx != 0:
                    p1 = p2 = p3 = p4 = p5 = p6 = None
                    reg1 = reg2 = reg3 = reg4 = reg5 = reg6 = 0

                    for tr in self.tracks:
                        (x, y) = tr[-1]
                        if 0 < x < x1 and 0 < y < y1:
                            reg1 += 1
                        if x2 < x < w and 0 < y < y1:
                            reg2 += 1
                        if 0 < x < x1 and y4 < y < h:
                            reg3 += 1
                        if x2 < x < w and y4 < y < h:
                            reg4 += 1
                        if 0 < x < x1 and y2 < y < y3:
                            reg5 += 1
                        if x2 < x < w and y2 < y < y3:
                            reg6 += 1

                    if reg1 < 10:
                        p1 = cv2.goodFeaturesToTrack(frame2, mask=mask1, **feature_params)
                    if reg2 < 10:
                        p2 = cv2.goodFeaturesToTrack(frame2, mask=mask2, **feature_params)
                    if reg3 < 10:
                        p3 = cv2.goodFeaturesToTrack(frame2, mask=mask3, **feature_params)
                    if reg4 < 10:
                        p4 = cv2.goodFeaturesToTrack(frame2, mask=mask4, **feature_params)
                    if reg5 < 10:
                        p5 = cv2.goodFeaturesToTrack(frame2, mask=mask5, **feature_params)
                    if reg6 < 10:
                        p6 = cv2.goodFeaturesToTrack(frame2, mask=mask6, **feature_params)

                # initialization(only runs at first frame)
                if self.frame_idx == 0:
                    p1 = cv2.goodFeaturesToTrack(frame2, mask=mask1, **feature_params)
                    p2 = cv2.goodFeaturesToTrack(frame2, mask=mask2, **feature_params)
                    p3 = cv2.goodFeaturesToTrack(frame2, mask=mask3, **feature_params)
                    p4 = cv2.goodFeaturesToTrack(frame2, mask=mask4, **feature_params)
                    p5 = cv2.goodFeaturesToTrack(frame2, mask=mask5, **feature_params)
                    p6 = cv2.goodFeaturesToTrack(frame2, mask=mask6, **feature_params)


                    for p in [p1, p2, p3, p4, p5, p6]:
                        if p is not None:
                            for x, y in p.reshape(-1, 2):
                                self.tracks.append(deque([(x, y)], maxlen=self.track_len))

                    initial_tracks = OpticalFlow(frame2, frame3, self.tracks, lk_params)
                    initial_src = np.float32([[list(tr[-2])] for tr in initial_tracks])
                    initial_dst = np.float32([[list(tr[-1])] for tr in initial_tracks])
                    HMat3to2, _ = cv2.findHomography(initial_src, initial_dst, 0, 5.0)

                # append found feature points
                for p in [p1, p2, p3, p4, p5, p6]:
                    if p is not None:
                        for x, y in p.reshape(-1, 2):
                            self.tracks.append(deque([(x, y)], maxlen=self.track_len))

            # iterate
            self.frame_idx += 1
            frame1 = frame2
            frame2 = frame3

            # cluster & track
            if self.frame_idx > 2:
                # find connected components
                nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thold1, None, None, None, 8, cv2.CV_32S)
                thold1 = cv2.cvtColor(thold1, cv2.COLOR_GRAY2BGR)

                # pick components with threshold
                centers = []
                if 0 < len(centroids) < 25:
                    ls = []
                    for c, s in zip(centroids, stats):
                        if 75 < s[4] < 5000:
                            c = tuple(c.astype(int))
                            sizewidth = int(s[2])
                            sizeheight = int(s[3])
                            ls.append((c, sizewidth, sizeheight))
                            # mark found components
                            #cv2.circle(thold1, c, 1, (0, 0, 255), 2)

                    # clustering
                    centers, sizels = clusterWithSize(ls, thresh=150)
                    for c in centers:
                        cv2.circle(thold1, c, 1, (0, 0, 255), 2)

                # tracking
                objs = tracker.update(centers)
                print(objs)
                
                merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)

                # Apply Kalman filter to tracking results
                for ID in tracker.objects_TF:
                    if tracker.objects_TF[ID] == True:
                        text = "ID {}".format(ID)
                        cent = objs[ID]
                        new = cent[-1]

                        if len(objs[ID]) == self.initiate_kalmanFilter:
                            kftext = "Kalman Filter Activated!!\n"
                            print(kftext)
                            kf = Kfilter()
                            kf.trainKfilter(cent)

                        if len(objs[ID]) > self.initiate_kalmanFilter:
                            kftext = "Kalman Filter Updating for object ID {}\n".format(ID)
                            print(kftext)
                            new = kf.updateKfilter(objs[ID][-1])
                            objs[ID][-1] = new
                            print(list(objs[ID]))

                        #print(cent)
                        #centx, centy = kalman_filter(cent)
                        #cv2.putText(vis, text, (cent[-1][0] - 10, cent[-1][1] - 10),
                        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        #cv2.circle(vis, (cent[-1][0], cent[-1][1]), 4, (0, 255, 0), -1)
                        #cv2.putText(vis, text, (int(centx[-1]) - 10, int(centy[-1]) - 10),
                        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        #cv2.circle(vis, (int(centx[-1]), int(centy[-1])), 4, (0, 255, 0), -1)
                        # visualize tracking results
                        cv2.putText(vis, text, (int(new[0]) - 10, int(new[1]) - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(vis, (int(new[0]), int(new[1])), 4, (0, 255, 0), -1)
                        cv2.putText(thold1, text, (int(new[0]) - 10, int(new[1]) - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(thold1, (int(new[0]), int(new[1])), 4, (0, 255, 0), -1)

                # draw
                final = np.hstack((vis, merged, thold1))
                cv2.imshow("frame", final)

            # waitkey
            k = cv2.waitKey(1) & 0xFF

            # interrupt
            if k == 27:
                print("User interrupt!")
                break

            # calculate FPS
            t_end = time.time()
            FPS = 1/(t_end-t_start+0.0001)
            totalFPS += FPS
            # print("FPS : ", "%.1f" % round(FPS, 3))


        # terminate
        self.vid.release()
        print("Average FPS :", round(totalFPS/self.frame_idx, 1))


a = App(video)
a.run()
cv2.destroyAllWindows()
