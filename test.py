import cv2
import imutils
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from imgops.subtract_imgs import subtract_images
from imgops.get_optflow import opticalflow

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict(maxCorners=7, qualityLevel=0.1, minDistance=7, blockSize=7, useHarrisDetector=False)
lk_params = dict(winSize=(9, 9), maxLevel=4, criteria=termination, minEigThreshold=1e-5)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())

if args["video"] == "cam":
    video = 0
else:
    video = args["video"]

np.set_printoptions(precision=3, suppress=True)


class App:
    def __init__(self):
        self.track_len = 5
        self.detect_interval = 2
        self.mask_size = 70
        self.tracks = []
        self.vid = cv2.VideoCapture(video)
        self.frame_idx = 0

    def run(self):
        # images for initialization
        ret, frame1 = self.vid.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame1 = imutils.resize(frame1, height=300)
        frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)

        ret, frame2 = self.vid.read()
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        frame2 = imutils.resize(frame2, height=300)
        frame2 = cv2.rotate(frame2, cv2.ROTATE_90_CLOCKWISE)

        # video size
        h, w = frame1.shape

        # mask for feature point search
        x1 = self.mask_size
        x2 = w - self.mask_size
        y1 = self.mask_size
        y2 = int(h/2 - self.mask_size/2)
        y3 = int(h/2 + self.mask_size/2)
        y4 = h - self.mask_size

        mask1 = np.zeros_like(frame1)
        mask1[0:y1, 0:x1] = 1
        mask2 = np.zeros_like(frame1)
        mask2[0:y1, x2:w] = 1
        mask3 = np.zeros_like(frame1)
        mask3[y4:h, 0:x1] = 1
        mask4 = np.zeros_like(frame1)
        mask4[y4:h, x2:w] = 1
        mask5 = np.zeros_like(frame1)
        mask5[y2:y3, 0:x1] = 1
        mask6 = np.zeros_like(frame1)
        mask6[y2:y3, x2:w] = 1

        # performance index variables
        totalFPS = 0
        HMatQidx = [[], []]
        samples = [np.array([[30], [30], [1]]), np.array([[30], [60], [1]]),
                   np.array([[60], [30], [1]]), np.array([[60], [60], [1]])]

        while True:
            t_start = time.time()

            # read and process frame
            ret, frame3 = self.vid.read()
            if not ret:
                print("End of video stream!")
                break

            frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
            frame3 = imutils.resize(frame3, height=300)
            frame3 = cv2.rotate(frame3, cv2.ROTATE_90_CLOCKWISE)

            # begin motion estimation
            if self.frame_idx > 0:
                img1, img2, img3 = frame1, frame2, frame3
                # histogram equalization for better visibility
                img1 = cv2.equalizeHist(img1)
                img2 = cv2.equalizeHist(img2)
                img3 = cv2.equalizeHist(img3)

                # optical flow from img2 to img3
                new_tracks = opticalflow(img2, img3, self.tracks, lk_params, track_length=self.track_len)

                # update track
                self.tracks = new_tracks

                # points in img3
                src23 = np.float32([[list(tr[-1])] for tr in self.tracks])
                # points in img2
                dst23 = np.float32([[list(tr[-2])] for tr in self.tracks])

                if len(dst23) >= 4:
                    # Homography Mat. that warps img1 to fit img2
                    HMat1to2 = np.linalg.inv(HMat3to2)
                    # Homography Mat. that warps img3 to fit img2
                    HMat3to2, stat = cv2.findHomography(src23, dst23, 0, 5.0)

                    print("Frame", self.frame_idx)
                    # print("Homography 1to2 :\n", HMat1to2)
                    # print("Homography 3to2 :\n", HMat3to2)
                    # print("\n")

                    # HMat quality index
                    total = 0
                    for s in samples:
                        total += np.linalg.norm(np.dot(HMat3to2, s) - s)
                    HMatQidx[0].append(self.frame_idx)
                    HMatQidx[1].append(np.sqrt(total))

                    # warping operation
                    warped1to2 = cv2.warpPerspective(img1, HMat1to2, (w, h))
                    warped3to2 = cv2.warpPerspective(img3, HMat3to2, (w, h))

                    # Gaussian blur operation to ease impact of edges
                    warped1to2 = cv2.GaussianBlur(warped1to2, (3, 3), 0)
                    warped3to2 = cv2.GaussianBlur(warped3to2, (3, 3), 0)
                    img2 = cv2.GaussianBlur(img2, (3, 3), 0)
                    # warped1to2 = cv2.medianBlur(warped1to2, 7)
                    # warped3to2 = cv2.medianBlur(warped3to2, 7)
                    # img2 = cv2.medianBlur(img2, 7)
                    # warped1to2 = cv2.blur(warped1to2, (2, 2))
                    # warped3to2 = cv2.blur(warped3to2, (2, 2))
                    # img2 = cv2.blur(img2, (2, 2))

                    # subtraction
                    # subt21 = subtract_images(img2, warped1to2, clip=0, isColor=False).astype('int32')
                    # subt23 = subtract_images(img2, warped3to2, clip=0, isColor=False).astype('int32')
                    subt21 = subtract_images(img2, warped1to2, clip=0, isColor=False)
                    subt23 = subtract_images(img2, warped3to2, clip=0, isColor=False)

                    # merge subtracted images
                    merged = (subt21 + subt23) / 2
                    # merged = np.hstack((subt21, subt23))
                    merged = merged.astype('uint8')

                    # crude thresholding
                    # _, merged = cv2.threshold(merged, 50, 255, cv2.THRESH_BINARY)

                    _, subt21 = cv2.threshold(subt21, 50, 255, cv2.THRESH_BINARY)
                    _, subt23 = cv2.threshold(subt23, 50, 255, cv2.THRESH_BINARY)
                    merged = cv2.bitwise_and(subt21, subt23)

                    # draw flow
                    merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
                    # cv2.polylines(merged, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

                # in case of motion compensation failure
                if len(dst23) < 4:
                    print("Motion Compensation Failure!")
                    subt21 = subtract_images(img2, img1, clip=0, isColor=False).astype('int32')
                    subt23 = subtract_images(img2, img3, clip=0, isColor=False).astype('int32')
                    merged = (subt21 + subt23) / 2
                    merged = merged.astype('uint8')
                    # merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
                    cv2.polylines(merged, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

            # search feature points
            if self.frame_idx % self.detect_interval == 0:
                if self.frame_idx != 0:
                    p1 = None
                    p2 = None
                    p3 = None
                    p4 = None
                    p5 = None
                    p6 = None
                    reg1, reg2, reg3, reg4, reg5, reg6 = 0, 0, 0, 0, 0, 0

                    for tr in self.tracks:
                        (x, y) = tr[-1]
                        if 0<x<x1 and 0<y<y1:
                            reg1 += 1
                        if x2<x<w and 0<y<y1:
                            reg2 += 1
                        if 0<x<x1 and y4<y<h:
                            reg3 += 1
                        if x2<x<w and y4<y<h:
                            reg4 += 1
                        if 0<x<x1 and y2<y<y3:
                            reg5 += 1
                        if x2<x<w and y2<y<y3:
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

            self.frame_idx += 1
            frame1 = frame2
            frame2 = frame3

            # draw image
            if self.frame_idx > 2:
                frame_draw = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
                frame_draw = frame_draw[10:h-10, 10:w-10]
                merged = merged[10:h - 10, 10:w - 10]
                merged = np.hstack((frame_draw, merged))
                merged = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
                # merged = cv2.rotate(merged, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # merged = imutils.resize(merged, width = 800)
                cv2.imshow("frame", merged)

            # exit
            k = cv2.waitKey(1) & 0xFF
            if k == ord("s"):
                initBB = cv2.selectROI("frame", merged, fromCenter=False, showCrosshair=True)
                (bbx, bby, bbw, bbh) = initBB
                print(merged[bby:bby+bbh, bbx:bbx+bbw])
                print(merged[bby:bby+bbh, 300+bbx:300+bbx+bbw])

            if k == 27:
                print("User interrupt!")
                break

            # calculate FPS
            t_end = time.time()
            FPS = 1/(t_end-t_start)
            totalFPS += FPS
            print("FPS : ", "%.1f" % round(FPS, 3))

        self.vid.release()
        print("Average FPS :", round(totalFPS/self.frame_idx, 1))

        # plot performance index
        plt.plot(HMatQidx[0], HMatQidx[1])
        plt.xlabel('Frame')
        plt.ylabel('Quality Index')
        plt.title('Homography Matrix')
        plt.ylim(0, 60)
        plt.show()


a = App()
a.run()
cv2.destroyAllWindows()
