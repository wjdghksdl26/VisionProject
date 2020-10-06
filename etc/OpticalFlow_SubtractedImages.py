import numpy as np
import cv2
import argparse
from imgops.subtract_imgs import subtract_images
from imgops.get_optflow import opticalflow

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict(maxCorners=30, qualityLevel=0.1, minDistance=7, blockSize=5)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=termination, minEigThreshold=1e-5)

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())

class App:
    def __init__(self, video_src=1):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.vid = cv2.VideoCapture(args["video"])
        self.frame_idx = 0
        self.blackscreen = False
        self.width = int(self.vid.get(3))
        self.height = int(self.vid.get(4))

    def run(self):
        while True:
            ret1, frame1 = self.vid.read() #프레임 받아오기
            ret2, frame2 = self.vid.read()
            if not ret1:
                break #다음 프레임 없으면 빠져나가기

            frame_gray = subtract_images(frame1, frame2, clip=0, width=0)

            vis = frame_gray.copy() #원래 frame 복사해둠(컬러 화면출력용)
            #vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            if len(self.tracks) > 0: #goodfeaturestotrack에서 뭐라도 찾아서 점들이 존재한다면
                img0, img1 = self.prev_gray, frame_gray #이전 프레임과 현재 프레임 갖고와서

                new_tracks = opticalflow(img0, img1, self.tracks, lk_params)

                self.tracks = new_tracks
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0)) #점들 이어서 선그리기

            if self.frame_idx % self.detect_interval == 0: #detect_interval마다 추적할 feature 찾기
                mask = np.ones_like(frame_gray) #goodfeaturestotrack에서 쓰는 변수 - 0인 점에서는 가동x
                p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params) #추적하기 좋은 점들을 출력(여러개)
                if p is not None: #뭐라도 찾았다면
                    for x, y in p.reshape(-1, 2): #찾은 점들 하나씩
                        self.tracks.append([(x,y)]) #self.tracks에 저장해두기
            self.frame_idx += 1 #frame세기
            self.prev_gray = frame_gray

            #kernel = np.ones((2, 2), np.uint8)
            #final = cv2.morphologyEx(final, cv2.MORPH_OPEN, kernel)
            #final = cv2.erode(vis, kernel, iterations=1)

            #final = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)

            cv2.imshow('frame', vis) #화면출력
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break

        self.vid.release()

video_src = 1
App(video_src).run()
cv2.destroyAllWindows()
