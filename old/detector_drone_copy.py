# packages for image processing
import cv2
import numpy as np
import time
from imgops.subtract_imgs import subtract_images
from imgops.get_optflow import opticalflow
from imgops.videostream import VideoStream

# packages for ROS integration
#import rospy
#from sensor_msgs.msg import Image
#from cv_bridge import CvBridge

# optical flow parameters
termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict(maxCorners=7, qualityLevel=0.03,
                      minDistance=7, blockSize=7, useHarrisDetector=False)
lk_params = dict(winSize=(15, 15), maxLevel=3,
                 criteria=termination, minEigThreshold=1e-5)


'''
# ROS node for processing video stream
rospy.init_node('detector_drone')
bridge = CvBridge()
image_pub = rospy.Publisher('~cv2', Image, queue_size=10)

def image_callback(data):
    cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')

    return cv_image
'''


class App:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 3
        self.mask_size = 70
        self.tracks = []
        self.vid = VideoStream(src=0).start()
        self.frame_idx = 0

    def run(self):
        # images for initialization
        frame1 = self.vid.read()
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        h, w = frame1.shape

        frame2 = self.vid.read()
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

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

        # kernel for morphology operations
        kernel = np.ones((3, 3))

        # main loop
        while True:
            try:
                t_start = time.time()

                # read and process frame
                frame3 = self.vid.read()

                # original frame
                vis = frame3.copy()
                frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

                # begin motion estimation
                if self.frame_idx > 0:
                    img1, img2, img3 = frame1, frame2, frame3

                    # optical flow from img2 to img3
                    new_tracks = opticalflow(
                        img2, img3, self.tracks, lk_params, track_length=self.track_len)

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
                        HMat3to2, stat = cv2.findHomography(
                            src23, dst23, 0, 5.0)

                        # current frame
                        print("Frame", self.frame_idx)

                        # warping operation
                        HMat1to2 = np.linalg.inv(HMat1to2)
                        warped1to2 = cv2.warpPerspective(
                            img1, HMat1to2, (w, h))
                        warped3to2 = cv2.warpPerspective(
                            img3, HMat3to2, (w, h))

                        # Gaussian blur operation to ease impact of edges
                        # parameter tuning required
                        warped1to2 = cv2.GaussianBlur(warped1to2, (3, 3), 0)
                        warped3to2 = cv2.GaussianBlur(warped3to2, (3, 3), 0)
                        img2 = cv2.GaussianBlur(img2, (3, 3), 0)

                        # subtracted images
                        subt21 = subtract_images(
                            img2, warped1to2, clip=20, isColor=False)
                        subt23 = subtract_images(
                            img2, warped3to2, clip=20, isColor=False)

                        # merge subtracted images
                        subt21 = subt21[20:h - 20, 20:w - 20]
                        subt23 = subt23[20:h - 20, 20:w - 20]
                        '''
                        subt21 = cv2.dilate(
                            subt21, kernel, iterations=3).astype('int32')
                        subt23 = cv2.dilate(
                            subt23, kernel, iterations=3).astype('int32')
                        '''
                        merged = (subt21 + subt23) / 2
                        subt21 = subt21.astype('uint8')
                        subt23 = subt23.astype('uint8')
                        merged = np.where(merged <= 50, 0, merged)
                        merged = merged.astype('uint8')

                        # ---------- essential operations finished ----------

                        # crude thresholding type 1
                        thold1 = merged.copy()
                        thold1 = cv2.erode(thold1, kernel, iterations=1)
                        _, thold1 = cv2.threshold(
                            thold1, 30, 255, cv2.THRESH_BINARY)
                        thold1 = cv2.dilate(thold1, kernel, iterations=5)

                        '''
                        # crude thresholding type 2
                        s21 = subt21.copy()
                        s23 = subt23.copy()
                        _, subt21 = cv2.threshold(subt21, 30, 255, cv2.THRESH_BINARY)
                        _, subt23 = cv2.threshold(subt23, 30, 255, cv2.THRESH_BINARY)
                        subt21 = cv2.erode(subt21, kernel, iterations=1)
                        subt23 = cv2.erode(subt23, kernel, iterations=1)
                        subt21 = cv2.dilate(subt21, kernel, iterations=3)
                        subt23 = cv2.dilate(subt23, kernel, iterations=3)
                        thold2 = cv2.bitwise_and(subt21, subt23)
                        thold2 = cv2.erode(thold2, kernel, iterations=1)
                        thold2 = cv2.dilate(thold2, kernel, iterations=3)
                        '''

                        # draw flow
                        for tr in self.tracks:
                            cv2.circle(vis, tuple(np.int32(tr[-1])), 2, (0, 0, 255), -1)
                        cv2.polylines(
                            vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))

                    # in case of motion compensation failure
                    if len(dst23) < 12:
                        print("Motion Compensation Failure!")
                        '''
                        subt21 = subtract_images(
                            img2, img1, clip=0, isColor=False)
                        subt23 = subtract_images(
                            img2, img3, clip=0, isColor=False)
                        subt21 = subt21[20:h - 20, 20:w - 20]
                        subt23 = subt23[20:h - 20, 20:w - 20]
                        subt21 = np.where(subt21 <= 30, 0, subt21)
                        subt23 = np.where(subt23 <= 30, 0, subt23)
                        merged = (subt21 + subt23) / 2
                        merged = merged.astype('uint8')
                        _, subt21 = cv2.threshold(
                            subt21, 30, 255, cv2.THRESH_BINARY)
                        _, subt23 = cv2.threshold(
                            subt23, 30, 255, cv2.THRESH_BINARY)
                        thold1 = cv2.bitwise_and(subt21, subt23)
                        '''
                        thold1 = np.zeros_like(img1)
                        thold1 = thold1.astype('uint8')
                        # thold2 = thold1

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

                        if reg1 < 7:
                            p1 = cv2.goodFeaturesToTrack(
                                frame2, mask=mask1, **feature_params)
                        if reg2 < 7:
                            p2 = cv2.goodFeaturesToTrack(
                                frame2, mask=mask2, **feature_params)
                        if reg3 < 7:
                            p3 = cv2.goodFeaturesToTrack(
                                frame2, mask=mask3, **feature_params)
                        if reg4 < 7:
                            p4 = cv2.goodFeaturesToTrack(
                                frame2, mask=mask4, **feature_params)
                        if reg5 < 7:
                            p5 = cv2.goodFeaturesToTrack(
                                frame2, mask=mask5, **feature_params)
                        if reg6 < 7:
                            p6 = cv2.goodFeaturesToTrack(
                                frame2, mask=mask6, **feature_params)

                    # initialization(only runs at first frame)
                    if self.frame_idx == 0:
                        p1 = cv2.goodFeaturesToTrack(
                            frame2, mask=mask1, **feature_params)
                        p2 = cv2.goodFeaturesToTrack(
                            frame2, mask=mask2, **feature_params)
                        p3 = cv2.goodFeaturesToTrack(
                            frame2, mask=mask3, **feature_params)
                        p4 = cv2.goodFeaturesToTrack(
                            frame2, mask=mask4, **feature_params)
                        p5 = cv2.goodFeaturesToTrack(
                            frame2, mask=mask5, **feature_params)
                        p6 = cv2.goodFeaturesToTrack(
                            frame2, mask=mask6, **feature_params)

                        for p in [p1, p2, p3, p4, p5, p6]:
                            if p is not None:
                                for x, y in p.reshape(-1, 2):
                                    self.tracks.append([(x, y)])

                        initial_tracks = opticalflow(
                            frame2, frame3, self.tracks, lk_params)
                        initial_src = np.float32(
                            [[list(tr[-2])] for tr in initial_tracks])
                        initial_dst = np.float32(
                            [[list(tr[-1])] for tr in initial_tracks])
                        HMat3to2, _ = cv2.findHomography(
                            initial_src, initial_dst, 0, 5.0)

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
                    vis = vis[20:h-20, 20:w-20]
                    merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
                    thold1 = cv2.cvtColor(thold1, cv2.COLOR_GRAY2BGR)
                    # thold2 = cv2.cvtColor(thold2, cv2.COLOR_GRAY2BGR)

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
                            ls.append(kpts[i].size)
                            if kpts[i].size > 20:
                                print(kpts[i].size)
                                print("Avoid!!")
                                cv2.putText(
                                    vis, "Avoid!!", (60, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                                # cv2.putText(vis, str(np.round_(ls[-1], 2)), (200, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                    vis = cv2.drawKeypoints(vis, kpts, np.array([]), (0, 0, 255),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    thold1 = cv2.drawKeypoints(thold1, kpts, np.array([]), (0, 0, 255),
                                               cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    final = np.hstack((vis, merged, thold1))
                    cv2.imshow("frame", final)

                    k = cv2.waitKey(1) & 0xFF

                    # interrupt
                    if k == 27:
                        print("User interrupt!")
                        self.vid.stop()
                        break

            except KeyboardInterrupt:
                self.vid.stop()
                break

            # calculate FPS
            t_end = time.time()
            FPS = 1/(t_end-t_start+0.0001)
            totalFPS += FPS
            # print("FPS : ", "%.1f" % round(FPS, 3))

        # terminate
        print("Average FPS :", round(totalFPS/self.frame_idx, 1))


a = App()
a.run()
