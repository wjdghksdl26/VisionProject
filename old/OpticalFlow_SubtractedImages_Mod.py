import numpy as np
import cv2
import imutils
import argparse
from imgops.subtract_imgs import subtract_images
from imgops.get_optflow import opticalflow

termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
feature_params = dict(maxCorners=15, qualityLevel=0.1, minDistance=10, blockSize=5)
lk_params = dict(winSize=(5, 5), maxLevel=2, criteria=termination, minEigThreshold=1e-5)

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
        self.detect_interval = 10
        self.tracks = []
        self.vid = cv2.VideoCapture(video)
        self.frame_idx = 0

    def run(self):
        while True:
            ret1, frame1 = self.vid.read() #프레임 받아오기
            frame1 = cv2.rotate(frame1, cv2.ROTATE_90_CLOCKWISE)
            if not ret1:
                break #다음 프레임 없으면 빠져나가기

            frame1 = imutils.resize(frame1, width=300)
            frame_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            h, w = frame_gray.shape

            vis = frame_gray.copy() #원래 frame 복사해둠(컬러 화면출력용)

            if len(self.tracks) > 0: #goodfeaturestotrack에서 뭐라도 찾아서 점들이 존재한다면
                img0, img1 = prev_gray, frame_gray #이전 프레임과 현재 프레임 갖고와서

                new_tracks = opticalflow(img0, img1, self.tracks, lk_params, track_length=self.track_len)
                self.tracks = new_tracks
                src = np.float32([[list(tr[-2])] for tr in self.tracks])
                dst = np.float32([[list(tr[-1])] for tr in self.tracks])
                if len(src) > 4:
                    H, status = cv2.findHomography(src, dst, 0, 5.0)
                    #H[0, 1] = 0
                    #H[1, 0] = 0
                    print("Homography - frame", self.frame_idx, "\n", H)
                    height, width = frame1.shape[:2]
                    warped = cv2.warpPerspective(img0, H, (width, height))

                    warped = cv2.GaussianBlur(warped, (5, 5), 0)
                    img1 = cv2.GaussianBlur(img1, (5, 5), 0)

                    vis = subtract_images(warped, img1, clip=0, isColor=False)
                    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))  # 점들 잇기

            if self.frame_idx % self.detect_interval == 0: #detect_interval마다 추적할 feature 찾기
                #mask = np.ones_like(frame_gray) #optical flow가 탐색할 영역 선택
                mask1 = np.zeros_like(frame_gray)
                mask1[0:100, 0:100] = 1
                mask2 = np.zeros_like(frame_gray)
                mask2[0:100, w-100:w] = 1
                mask3 = np.zeros_like(frame_gray)
                mask3[h-100:h, 0:100] = 1
                mask4 = np.zeros_like(frame_gray)
                mask4[h-100:h, w-100:w] = 1
                mask5 = np.zeros_like(frame_gray)
                mask5[int(h/2 - 50):int(h/2 + 50), 0:100] = 1
                mask6 = np.zeros_like(frame_gray)
                mask6[int(h/2 - 50):int(h/2 + 50), w - 100:w] = 1

                #p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params) #추적하기 좋은 점들을 출력(여러개)
                p1 = cv2.goodFeaturesToTrack(frame_gray, mask=mask1, **feature_params)
                p2 = cv2.goodFeaturesToTrack(frame_gray, mask=mask2, **feature_params)
                p3 = cv2.goodFeaturesToTrack(frame_gray, mask=mask3, **feature_params)
                p4 = cv2.goodFeaturesToTrack(frame_gray, mask=mask4, **feature_params)
                p5 = cv2.goodFeaturesToTrack(frame_gray, mask=mask5, **feature_params)
                p6 = cv2.goodFeaturesToTrack(frame_gray, mask=mask6, **feature_params)
                p = [[[0, 0]]]
                for i in [p1, p2, p3, p4, p5, p6]:
                    if i is not None:
                        p = np.vstack((p, i))

                if p is not None: #뭐라도 찾았다면
                    for x, y in p.reshape(-1, 2): #찾은 점들 하나씩
                        self.tracks.append([(x,y)]) #self.tracks에 저장해두기
            self.frame_idx += 1 #frame세기
            prev_gray = frame_gray #이전 frame으로 저장
            if self.frame_idx > 1:
                #vis = vis[10:h-10, 10:w-10]
                #vis = imutils.resize(vis, width=600)
                cv2.imshow('frame', vis)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                print("User interrupt!")
                break

        self.vid.release()

a = App()
a.run()
cv2.destroyAllWindows()
