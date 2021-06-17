import copy
import cv2
import imutils
import pyrealsense2 as rs
import numpy as np
import argparse
import time

from logicops.tracker import Tracker
from logicops.cluster import clusterWithSize

class Detector:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        #self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 60)
        self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 60)
        self.config.enable_stream(rs.stream.infrared, 1, 424, 240, rs.format.y8, 60)
        #self.align = rs.align(rs.stream.color)
        self.tracker = Tracker()

    def initialize(self):
        self.pipeline.start(self.config)
        data = self.pipeline.wait_for_frames()
        #color = data.get_color_frame()
        depth = data.get_depth_frame()
        depth = np.asanyarray(depth.get_data())
        h, w = depth.shape

        return h, w

    def run(self):
        h, w = self.initialize()

        xlist = list(range(30, w-30, 15))
        ylist = list(range(30, h-30, 15))
        depthMat_zero = np.zeros((len(ylist), len(xlist)))

        while True:
            ts = time.time()
            data = self.pipeline.wait_for_frames()
            ir_frame = data.first(rs.stream.infrared)
            ir_frame = np.asanyarray(ir_frame.get_data())
            #data = self.align.process(data)
            #color = data.get_color_frame()
            #frame = np.asanyarray(color.get_data())
            depth = data.get_depth_frame()

            depthMat = depthMat_zero
            for xidx, x in enumerate(xlist):
                for yidx, y in enumerate(ylist):
                    dist = round(depth.get_distance(x, y), 2)
                    depthMat[yidx, xidx] = dist
                    if 0.3 < dist < 1.0:
                        depthMat[yidx, xidx] = 255 - dist * 255
                        #cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

            depthMat = depthMat.astype('uint8')
            _, depthMatThr = cv2.threshold(imutils.resize(depthMat, height = 60), 10, 255, cv2.THRESH_BINARY)
            
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(depthMatThr, None, None, None, 8, cv2.CV_32S)
            stats = stats[1:]
            centroids = centroids[1:]

            centers = []
            if 0 < len(centroids) < 10:
                ls = []
                for c, s in zip(centroids, stats):
                    if 25 < s[4]:
                        c = tuple(c)
                        sizewidth = float(s[2])
                        sizeheight = float(s[3])
                        ls.append((c, sizewidth, sizeheight))

                centers, sizels = clusterWithSize(ls, thresh=20.0)

            objs = self.tracker.update(centers)

            ir_frame = ir_frame[30:h-30, 30:w-30]
            ir_frame = imutils.resize(ir_frame, height=300)
            depthMat = imutils.resize(depthMat, height=300)
            ir_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
            depthMat = cv2.cvtColor(depthMat, cv2.COLOR_GRAY2BGR)
            
            for ID in self.tracker.objects_TF:
                if self.tracker.objects_TF[ID] == True:
                    text = "ID {}".format(ID)
                    cent = objs[ID]
                    new = cent[-1]

                    cv2.putText(ir_frame, text, (int(new[0])*5-10, int(new[1])*5-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(ir_frame, (int(new[0])*5, int(new[1])*5), 4, (0, 255, 0), -1)
                    cv2.putText(depthMat, text, (int(new[0])*5-10, int(new[1])*5-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(depthMat, (int(new[0])*5, int(new[1])*5), 4, (0, 255, 0), -1)


            img = np.hstack((ir_frame, depthMat))
            cv2.imshow("frame", img)
            k = cv2.waitKey(1) & 0xFF

            if k == 27:
                print("User interrupt!")
                break
            te = time.time()
            t = te - ts
            fps = 1 / t
            print("FPS :", fps)

        cv2.destroyAllWindows()

detector = Detector()
detector.run()