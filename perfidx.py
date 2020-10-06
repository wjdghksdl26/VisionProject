import argparse
import cv2
import matplotlib.pyplot as plt
from test2 import App
from main import BGSubt
from Improved_OptFlow_Main import opticalflow_subtractedimgs

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str)
args = vars(ap.parse_args())

videoPath = args["video"]

a=App(videoPath)
frame1, HIdx1, noise1, fp1 = a.run()
cv2.destroyAllWindows()
b=BGSubt(videoPath)
frame2, HIdx2, noise2, fp2 = b.bgsubtraction()
cv2.destroyAllWindows()
frame3, HIdx3, noise3, fp3=opticalflow_subtractedimgs(videoPath)
cv2.destroyAllWindows()

plt.rc('font', size = 18)
plt.plot(frame3, HIdx3)
plt.plot(frame2, HIdx2)
plt.plot(frame1, HIdx1)
plt.legend(['Existing Works', 'Previous System', 'Revised System'])
plt.xlabel('Frame')
plt.ylabel('HMat Index')
plt.title('Homography Matrix Index')
plt.ylim(0, 100)
plt.show()

plt.close()

plt.plot(frame3, noise3)
plt.plot(frame2, noise2)
plt.plot(frame1, noise1)
plt.legend(['Existing Works', 'Previous System', 'Revised System'])
plt.xlabel('Frame')
plt.ylabel('Noise')
plt.title('Noise Per Frame')
plt.show()

plt.close()

plt.plot(frame3, fp3)
plt.plot(frame2, fp2)
plt.plot(frame1, fp1)
plt.legend(['Existing Works', 'Previous System', 'Revised System'])
plt.xlabel('Frame')
plt.ylabel('Feature Points')
plt.title('Feature Points Per Frame')
plt.show()