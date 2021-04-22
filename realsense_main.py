import cv2
import pyrealsense2.pyrealsense2 as rs
import imutils
import numpy as np
import argparse
import time

from imgops.get_optflow_test import OpticalFlow
from imgops.videostream import VideoStream

from logicops.cluster import clusterWithSize
from logicops.tracker import Tracker
from logicops.kalman2 import Kfilter
from logicops.count import count

import rospy
from geometry_msgs.msg import Point

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())
source = args["video"]


class App:
    def __init__(self, source="realsense"):
        self.source = source
        if source == "realsense":
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
            self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30)
            self.pipeline.start(self.config)
        elif source == "cam":
            self.video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.video.set(cv2.CAP_PROP_FPS, 60)
        else:
            self.video = cv2.VideoCapture(source)
        
        self.detect_interval = 2
        self.mask_size = 70
        self.kernel = np.ones((3, 3))
        #self.initiate_kalmanFilter = 12
        self.termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        self.feature_params = dict(maxCorners=10, qualityLevel=0.01, minDistance=3, blockSize=7, useHarrisDetector=True)
        self.lk_params = dict(winSize=(35, 35), maxLevel=2, criteria=self.termination, minEigThreshold=1e-4)
        self.tracker = Tracker()
        
        print("Starting ROS Node for Publishing Object Coordinates...")
        self.pub = rospy.Publisher('/moving_obj_coords', Point, queue_size = 1)
        rospy.init_node('detector')

    def run(self):
        if self.source == "realsense":
            for i in range(5):
                print("Initializing RealSense Camera...")
                temp = self.pipeline.wait_for_frames()
                time.sleep(0.5)
                
            print("RealSense Initialization Complete!!")
                
            frame = self.pipeline.wait_for_frames()
            frame1 = frame.get_color_frame()
            frame1 = np.asanyarray(frame1.get_data())
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame = self.pipeline.wait_for_frames()
            frame2 = frame.get_color_frame()
            depth2 = frame.get_depth_frame()
            frame2 = np.asanyarray(frame2.get_data())
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            vid = self.video
            ret, frame1 = vid.read()
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            if self.source != "cam":
                frame1 = imutils.resize(frame1, width=320)

            ret, frame2 = vid.read()
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            if self.source != "cam":
                frame2 = imutils.resize(frame2, width=320)

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

        frame_idx = 0
        totalFPS = 0.0
        while True:
            t_start = time.time()
            print("Frame", frame_idx)

            if self.source == "realsense":
                frame = self.pipeline.wait_for_frames()
                frame3 = frame.get_color_frame()
                depth3 = frame.get_depth_frame()
                vis = np.asanyarray(frame3.get_data())
                frame3 = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
            
            else:
                ret, frame3 = vid.read()
                if not ret:
                    print("End of stream!!")
                    break
                if self.source != "cam":
                    frame3 = imutils.resize(frame3, width=320)
                
                vis = frame3.copy()
                frame3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)

            vis = vis[15:h-15, 15:w-15]

            if frame_idx > 0:
                img1, img2, img3 = frame1, frame2, frame3

                src23, dst23 = OpticalFlow(img2, img3, dst23, self.lk_params)

                if dst23 is not None and len(dst23) >= 20 :
                    # Homography Mat. that warps img1 to fit img2
                    HMat1to2 = HMat3to2
                    # Homography Mat. that warps img3 to fit img2
                    HMat3to2, stat = cv2.findHomography(dst23, src23, cv2.RANSAC, 1.0)

                    # warping operation (high load)
                    HMat1to2 = np.linalg.inv(HMat1to2)
                    warped1to2 = cv2.warpPerspective(img1, HMat1to2, (w, h))
                    # OpenCV 3.x does not have cv2.WARP_INVERSE_MAP
                    # warped1to2 = cv2.warpPerspective(img1, HMat1to2, (w, h), cv2.INTER_LINEAR, cv2.WARP_INVERSE_MAP)
                    warped3to2 = cv2.warpPerspective(img3, HMat3to2, (w, h))

                    # subtracted images
                    subt21 = np.clip(cv2.subtract(warped1to2, img2), 15, None)
                    subt23 = np.clip(cv2.subtract(warped3to2, img2), 15, None)

                    # merge subtracted images
                    subt21 = subt21[15:h - 15, 15:w - 15]
                    # subt21 = cv2.blur(subt21, (3, 3))
                    subt23 = subt23[15:h - 15, 15:w - 15]
                    # subt23 = cv2.blur(subt21, (3, 3))
                    merged = ((subt21 + subt23) / 2).astype('uint8')
                    merged = cv2.blur(merged, (5, 5))

                    # ---------- essential operations finished ----------

                    # thresholding
                    thold1 = merged.copy()
                    #thold1 = cv2.erode(thold1, self.kernel, iterations=1)
                    _, thold1 = cv2.threshold(thold1, 40, 255, cv2.THRESH_BINARY)
                    thold1 = cv2.dilate(thold1, self.kernel, iterations=2)

                    # draw flow
                    for tr in dst23:
                            cv2.circle(vis, tuple(np.int32(tr[0])), 2, (0, 0, 255), -1)

                # in case of motion compensation failure
                if dst23 is None or len(dst23) < 20:
                    print("Motion Compensation Failure!")
                    thold1 = np.zeros_like(img1)
                    thold1 = thold1[15:h-15, 15:w-15]
                    thold1 = thold1.astype('uint8')
                    merged = thold1

            # search feature points
            if frame_idx % self.detect_interval == 0:
                # after initialization
                if frame_idx != 0:
                    p1 = p2 = p3 = p4 = p5 = p6 = None

                    # cython boost
                    if dst23 is not None:
                        dst = np.asarray(dst23, dtype=int).reshape(-1, 2)
                        reg1, reg2, reg3, reg4, reg5, reg6 = count(dst, x1, x2, y1, y2, y3, y4, w, h)

                    plist = [dst23]
                    if reg1:
                        p1 = cv2.goodFeaturesToTrack(frame3, mask=mask1, **self.feature_params)
                        plist.append(p1)
                    if reg2:
                        p2 = cv2.goodFeaturesToTrack(frame3, mask=mask2, **self.feature_params)
                        plist.append(p2)
                    if reg3:
                        p3 = cv2.goodFeaturesToTrack(frame3, mask=mask3, **self.feature_params)
                        plist.append(p3)
                    if reg4:
                        p4 = cv2.goodFeaturesToTrack(frame3, mask=mask4, **self.feature_params)
                        plist.append(p4)
                    if reg5:
                        p5 = cv2.goodFeaturesToTrack(frame3, mask=mask5, **self.feature_params)
                        plist.append(p5)
                    if reg6:
                        p6 = cv2.goodFeaturesToTrack(frame3, mask=mask6, **self.feature_params)
                        plist.append(p6)

                    # append found feature points
                    plist = [i for i in plist if i is not None]
                    dst23 = np.concatenate(plist, axis=0)

                # initialization(only runs at first frame)
                if frame_idx == 0:
                    plist = []
                    p1 = cv2.goodFeaturesToTrack(frame3, mask=mask1, **self.feature_params)
                    plist.append(p1)
                    p2 = cv2.goodFeaturesToTrack(frame3, mask=mask2, **self.feature_params)
                    plist.append(p2)
                    p3 = cv2.goodFeaturesToTrack(frame3, mask=mask3, **self.feature_params)
                    plist.append(p3)
                    p4 = cv2.goodFeaturesToTrack(frame3, mask=mask4, **self.feature_params)
                    plist.append(p4)
                    p5 = cv2.goodFeaturesToTrack(frame3, mask=mask5, **self.feature_params)
                    plist.append(p5)
                    p6 = cv2.goodFeaturesToTrack(frame3, mask=mask6, **self.feature_params)
                    plist.append(p6)

                    plist = [i for i in plist if i is not None]
                    print(plist)
                    dst23 = np.concatenate(plist, axis=0)

                    src23, dst23 = OpticalFlow(frame2, frame3, dst23, self.lk_params)
                    HMat3to2, _ = cv2.findHomography(src23, dst23, 0, 1.0)

            # iterate
            frame_idx += 1
            frame1 = frame2
            frame2 = frame3

            # cluster & track
            if frame_idx > 2:
                # find connected components
                nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thold1, None, None, None, 8, cv2.CV_32S)
                thold1 = cv2.cvtColor(thold1, cv2.COLOR_GRAY2BGR)
                centers = []

                # pick components with threshold
                if 0 < len(centroids) < 25:
                    centers = []
                    ls = []
                    for c, s in zip(centroids, stats):
                        if 25 < s[4] < 5000:
                            c = tuple(c)
                            sizewidth = float(s[2])
                            sizeheight = float(s[3])
                            ls.append((c, sizewidth, sizeheight))
                            # mark found components
                            # cv2.circle(thold1, c, 1, (0, 0, 255), 2)

                    # clustering
                    centers, sizels = clusterWithSize(ls, thresh=150.0)
                    # for c in centers:
                    #     cv2.circle(thold1, (int(c[0]), int(c[1])), 1, (0, 0, 255), 2)

                # tracking
                objs = self.tracker.update(centers)
                print(objs)
                
                merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)

                for ID in self.tracker.objects_TF:
                    if self.tracker.objects_TF[ID] == True:
                        text = "ID {}".format(ID)
                        cent = objs[ID]
                        new = cent[-1]

                        cv2.putText(vis, text, (int(new[0]) - 10, int(new[1]) - 10),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.circle(vis, (int(new[0]), int(new[1])), 4, (0, 255, 0), -1)

                        # get distance to object
                        if self.source == "realsense":
                            xlist = list(range(int(new[0]) - 20, int(new[0]) + 21, 5))
                            ylist = list(range(int(new[1]) - 20, int(new[1]) + 21, 5))
                            dist = 100.0
                            for i in xlist:
                                if i < 0 or i > 480:
                                    continue
                                for j in ylist:
                                    if j < 0 or j > 240:
                                        continue
                                    ndist = round(depth2.get_distance(i, j), 2)
                                    if 0.2 < ndist < dist:
                                        dist = ndist
                            cv2.putText(vis, "dist "+str(dist)+" m", (int(new[0]) - 10, int(new[1]) + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # publish via ROS
                            point = Point()
                            point.x = new[0]
                            point.y = new[1]
                            point.z = dist
                            self.pub.publish(point)
                
                depth2 = depth3
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
        if self.source == "cam":
            vid.release()
        print("Average FPS :", round(totalFPS / frame_idx, 1))


if __name__ == "__main__":
    a = App(source)
    a.run()
    cv2.destroyAllWindows()
