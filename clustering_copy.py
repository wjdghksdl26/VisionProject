import cv2
import imutils
import numpy as np
import argparse
import time
from imgops.subtract_imgs import subtract_images
from imgops.get_optflow import opticalflow
from imgops.videostream import VideoStream
from logicops.cluster import clusterwithsize
from logicops.tracker import Tracker

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict(maxCorners=15, qualityLevel=0.1, minDistance=3, blockSize=7, useHarrisDetector=False)
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
        self.mask_size = 50
        self.tracks = []
        self.vid = videoPath
        self.frame_idx = 0
        self.rotate = cv2.ROTATE_90_CLOCKWISE

    def run(self):
        # images for initialization
        ret, frame1 = self.vid.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame1 = imutils.resize(frame1, width=300)
        #frame1 = cv2.rotate(frame1, self.rotate)

        ret, frame2 = self.vid.read()
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2 = imutils.resize(frame2, width=300)
        #frame2 = cv2.rotate(frame2, self.rotate)

        # video size
        h, w = frame1.shape

        # mask for feature point search
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

        #kernel for morphology operations
        kernel = np.ones((3, 3))
        kernel5 = np.ones((5, 5))
        gaussiankernel = (3, 3)

        tracker = Tracker()

        # main loop
        while True:
            t_start = time.time()

            # read and process frame
            ret, frame3 = self.vid.read()
            if not ret:
                print("End of video stream!")
                break

            # original frame
            vis = frame3.copy()
            frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
            frame3 = imutils.resize(frame3, width=300)
            #frame3 = cv2.rotate(frame3, self.rotate)

            vis = imutils.resize(vis, width=300)
            #vis = cv2.rotate(vis, self.rotate)

            # begin motion estimation
            if self.frame_idx > 0:
                img1, img2, img3 = frame1, frame2, frame3

                # optical flow from img2 to img3
                new_tracks = opticalflow(img2, img3, self.tracks, lk_params, track_length=self.track_len)

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
                    warped1to2 = cv2.warpPerspective(img1, HMat1to2, (w, h), cv2.INTER_LINEAR, cv2.WARP_INVERSE_MAP)
                    warped3to2 = cv2.warpPerspective(img3, HMat3to2, (w, h), cv2.INTER_LINEAR)

                    # Gaussian blur operation to ease impact of edges
                    # parameter tuning required
                    #warped1to2 = cv2.GaussianBlur(warped1to2, gaussiankernel, 0)
                    #warped3to2 = cv2.GaussianBlur(warped3to2, gaussiankernel, 0)
                    #img2 = cv2.GaussianBlur(img2, gaussiankernel, 0)

                    # subtracted images
                    subt21 = subtract_images(warped1to2, img2, clip=15, isColor=False)
                    subt23 = subtract_images(warped3to2, img2, clip=15, isColor=False)

                    # merge subtracted images
                    subt21 = subt21[15:h - 15, 15:w - 15]
                    subt23 = subt23[15:h - 15, 15:w - 15]
                    subt21 = cv2.medianBlur(subt21, 3)
                    subt23 = cv2.medianBlur(subt23, 3)
                    subt21 = cv2.erode(subt21, kernel)
                    subt23 = cv2.erode(subt23, kernel)
                    subt21 = cv2.dilate(subt21, kernel, iterations=4).astype('int32')
                    subt23 = cv2.dilate(subt23, kernel, iterations=4).astype('int32')
                    merged = (subt21 + subt23) / 2
                    merged = np.where(merged <= 50, 0, merged)
                    merged = merged.astype('uint8')
                    merged = merged * 2
                    #merged = cv2.dilate(merged, kernel, iterations=1)

                    # ---------- essential operations finished ----------

                    # crude thresholding type 1
                    thold1 = merged.copy()
                    #thold1 = cv2.erode(thold1, kernel, iterations=2)
                    _, thold1 = cv2.threshold(thold1, 60, 255, cv2.THRESH_BINARY)
                    thold1 = cv2.dilate(thold1, kernel, iterations=2)

                    # draw flow
                    merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
                    cv2.polylines(merged, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

                # in case of motion compensation failure
                if len(dst23) < 12:
                    print("Motion Compensation Failure!")
                    thold1 = np.zeros_like(img1)
                    thold1 = thold1[15:h-15, 15:w-15]
                    thold1 = thold1.astype('uint8')

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

                    if reg1 < 15:
                        p1 = cv2.goodFeaturesToTrack(frame2, mask=mask1, **feature_params)
                    if reg2 < 15:
                        p2 = cv2.goodFeaturesToTrack(frame2, mask=mask2, **feature_params)
                    if reg3 < 15:
                        p3 = cv2.goodFeaturesToTrack(frame2, mask=mask3, **feature_params)
                    if reg4 < 15:
                        p4 = cv2.goodFeaturesToTrack(frame2, mask=mask4, **feature_params)
                    #if reg5 < 15:
                    #    p5 = cv2.goodFeaturesToTrack(frame2, mask=mask5, **feature_params)
                    #if reg6 < 15:
                    #    p6 = cv2.goodFeaturesToTrack(frame2, mask=mask6, **feature_params)

                # initialization(only runs at first frame)
                if self.frame_idx == 0:
                    p1 = cv2.goodFeaturesToTrack(frame2, mask=mask1, **feature_params)
                    p2 = cv2.goodFeaturesToTrack(frame2, mask=mask2, **feature_params)
                    p3 = cv2.goodFeaturesToTrack(frame2, mask=mask3, **feature_params)
                    p4 = cv2.goodFeaturesToTrack(frame2, mask=mask4, **feature_params)
                    #p5 = cv2.goodFeaturesToTrack(frame2, mask=mask5, **feature_params)
                    #p6 = cv2.goodFeaturesToTrack(frame2, mask=mask6, **feature_params)
                    p5 = None
                    p6 = None

                    for p in [p1, p2, p3, p4, p5, p6]:
                        if p is not None:
                            for x, y in p.reshape(-1, 2):
                                self.tracks.append([(x, y)])

                    initial_tracks = opticalflow(frame2, frame3, self.tracks, lk_params)
                    initial_src = np.float32([[list(tr[-2])] for tr in initial_tracks])
                    initial_dst = np.float32([[list(tr[-1])] for tr in initial_tracks])
                    HMat3to2, _ = cv2.findHomography(initial_src, initial_dst, 0, 5.0)

                # append found feature points
                for p in [p1, p2, p3, p4, p5, p6]:
                    if p is not None:
                        for x, y in p.reshape(-1, 2):
                            self.tracks.append([(x, y)])

            # iterate
            self.frame_idx += 1
            frame1 = frame2
            frame2 = frame3

            # draw image
            if self.frame_idx > 2:
                #merged = merged[20:h - 20, 20:w - 20]
                vis = vis[15:h-15, 15:w-15]
                nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thold1, None, None, None, 8, cv2.CV_32S)
                thold1 = cv2.cvtColor(thold1, cv2.COLOR_GRAY2BGR)

                centers = []
                if 1 < len(centroids) < 7:
                    ls = []
                    for c, s in zip(centroids, stats):
                        if 10 < s[4] < 5000:
                            c = tuple(c.astype(int))
                            sz = int(s[4])
                            ls.append((c, sz))
                            cv2.circle(thold1, c, 1, (0, 0, 255), 2)

                    centers, sizels = clusterwithsize(ls, thresh=100)
                    print(len(centroids))
                    #print(centers)
                    
                    #if len(centers) < 10:
                        #for idx, i in enumerate(centers):
                            #cv2.circle(vis, tuple(i), int(np.sqrt(sizels[idx])), (0, 0, 255), 2)
                            

                objs = tracker.update(centers)
                for (ID, cent) in objs.items():
                    if len(cent) > 3:
                        text = "ID {}".format(ID)
                        cv2.putText(vis, text, (cent[-1][0] - 10, cent[-1][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(vis, (cent[-1][0], cent[-1][1]), 4, (0, 255, 0), -1)
                
                print(objs)


                '''
                thold1 = cv2.cvtColor(thold1, cv2.COLOR_GRAY2BGR)
                
                kpt = cv2.cvtColor(thold1, cv2.COLOR_BGR2GRAY)
                kpt_inv = cv2.bitwise_not(kpt)
                params = cv2.SimpleBlobDetector_Params()
                params.minThreshold = 0
                params.maxThreshold = 255
                params.filterByArea = True
                params.minArea = 15
                params.filterByInertia = False
                params.minInertiaRatio = 0.1
                params.filterByColor = False
                params.blobColor = 0
                params.filterByCircularity = False
                params.filterByConvexity = False
                params.minConvexity = 0.5
                detector = cv2.SimpleBlobDetector_create(params)
                kpts = detector.detect(kpt_inv)
                
                
                if len(kpts) > 0:
                    ls = []
                    for i in range(len(kpts)):
                        ls.append((kpts[i].pt, kpts[i].size))
                        if kpts[i].size > 20:
                            print("Avoid!!")
                            cv2.putText(vis, "Avoid!!", (60, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                            # cv2.putText(vis, str(np.round_(ls[-1], 2)), (200, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    print(ls)
                

                if len(kpts) > 0:
                    ls = []
                    for i in kpts:
                        ls.append(i.pt)
                
                    centers = cluster(ls, thresh=30)
                    print(centers)
                

                vis = cv2.drawKeypoints(vis, kpts, np.array([]), (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                thold1 = cv2.drawKeypoints(thold1, kpts, np.array([]), (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                '''
                

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
