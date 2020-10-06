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


def opticalflow_subtractedimgs(videoPath=0):
    vid = cv2.VideoCapture(videoPath)
    frame_idx = 0
    totalTime = 0
    tracks = []

    HMatQidx = [[], []]
    samples = [np.array([[30], [30], [1]]), np.array([[30], [60], [1]]),
               np.array([[60], [30], [1]]), np.array([[60], [60], [1]])]
    noiselist = []
    fplist = []

    while 1:
        t_start = time.time()
        print("Frame", frame_idx)
        ret, frame = vid.read()

        if not ret:
            print("End of Video Stream!")
            break

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = imutils.resize(frame, width = 300)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = frame_gray.shape

        if frame_idx > 0:  # at least two image are required for subtraction
            frame_subt_gray = subtract_images(prev_gray, frame_gray, clip=0, isColor=False)

            if frame_idx > 1:  # at least three images are required to compute optical flow between subtracted imgs
                new_tracks = opticalflow(prev_subt_gray, frame_subt_gray, tracks, lk_params, track_length=track_len)

                if len(new_tracks) == 0:  # optical flow failure
                    vis = subtract_images(prev_subt_gray, frame_subt_gray, isColor=False)  # just show subtracted imgs
                    vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

                if len(new_tracks) != 0:
                    tracks = new_tracks
                    src = np.float32([[list(tr[-2])] for tr in tracks])
                    dst = np.float32([[list(tr[-1])] for tr in tracks])

                    if len(src) > 3:  # necessary conditions for homography calculation
                        H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

                        total = 0
                        for s in samples:
                            total += np.linalg.norm(np.dot(H, s) - s)

                        HMatQidx[0].append(frame_idx)
                        HMatQidx[1].append(np.sqrt(total))

                        # warped = cv2.warpPerspective(prev_subt_gray, H, (width, height))
                        # warped = prev_subt_gray
                        warped = cv2.warpPerspective(prev_gray, H, (w, h))
                        # vis = subtract_images(warped, frame_subt_gray, clip=0, isColor=False)
                        vis = subtract_images(warped, frame_gray, clip=0, isColor=False)

                        _, thold = cv2.threshold(vis, 50, 255, cv2.THRESH_BINARY)
                        thold = cv2.cvtColor(thold, cv2.COLOR_GRAY2BGR)

                        noisecalc = vis[20:h - 20, 20:w - 20]
                        noise = np.sum(noisecalc - 50)
                        noiselist.append(noise)
                        fplist.append(len(tracks))

                        # draw optical flow
                        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
                        cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

                    if len(src) < 4:
                        print("Motion compensation failure!")
                        vis = subtract_images(prev_gray, frame_gray, clip=0, isColor=False)
                        _, thold = cv2.threshold(vis, 50, 255, cv2.THRESH_BINARY)
                        thold = cv2.cvtColor(thold, cv2.COLOR_GRAY2BGR)
                        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

            prev_subt_gray = frame_subt_gray

        prev_gray = frame_gray

        if frame_idx % detect_interval == 1:  # to find feature pts of a subtracted image, two images are required
            mask = np.ones_like(frame_gray)
            #mask = np.zeros_like(frame_gray)
            #mask[:, 0:100] = 1
            #mask[:, 500:] = 1
            p = cv2.goodFeaturesToTrack(prev_subt_gray, mask=mask, **feature_params)
            if p is not None:
                for x, y in p.reshape(-1, 2):
                    tracks.append([(x, y)])

        if frame_idx > 1:
            #kernel = np.ones((2, 2), np.uint8)
            #vis = cv2.morphologyEx(vis, cv2.MORPH_OPEN, kernel)
            vis = np.hstack((frame, thold))
            # vis = cv2.rotate(vis, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imshow('result', vis)
        frame_idx += 1
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            print("User interrupt!")
            break

        t_end = time.time()
        print("Elapsed time:", "%.3f" % round(t_end-t_start, 3), "sec\n")
        totalTime = totalTime + (t_end - t_start)

    print("Avg. processing time per frame =", round(totalTime / frame_idx, 3), "sec\n")

    vid.release()

    plt.subplot(131)
    plt.plot(HMatQidx[0], HMatQidx[1])
    #ax = plt.gca()
    #ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel('Frame')
    plt.title('Homography Matrix')
    plt.ylim(0, 100)

    plt.subplot(132)
    plt.plot(HMatQidx[0], noiselist)
    #ax = plt.gca()
    #ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel('Frame')
    plt.title('Total Noise - Lower is better')
    # plt.ylim(0, 100000)

    plt.subplot(133)
    plt.plot(HMatQidx[0], fplist)
    #ax = plt.gca()
    #ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.xlabel('Frame')
    plt.title('No. of Feature Points')
    plt.ylim(0, 100)

    plt.show()

    return HMatQidx[0], HMatQidx[1], noiselist, fplist










def draw_of_subtractedimgs(videoPath=0):
    vid = cv2.VideoCapture(videoPath)
    frame_idx = 0
    totalTime = 0
    tracks = []

    while 1:
        t_start = time.time()
        ret, frame = vid.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        if not ret:
            print("End of Video!")
            break

        frame = imutils.resize(frame, width=600)
        print("Frame", frame_idx)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frame_idx > 0:  # at least two image are required for subtraction
            frame_subt_gray = subtract_images(prev_gray, frame_gray, clip=0, absolute=False)
            vis = cv2.cvtColor(frame_subt_gray, cv2.COLOR_GRAY2BGR).copy()

            if frame_idx > 1:  # at least three images are required to compute optical flow between subtracted imgs
                new_tracks = opticalflow(prev_subt_gray, frame_subt_gray, tracks, lk_params, track_length=track_len)

                if len(new_tracks) != 0:
                    tracks = new_tracks
                    for tr in tracks:
                        cv2.circle(vis, tuple(tr[-1]), 3, (0, 0, 255), -1)
                    cv2.polylines(vis, [np.int32(tr) for tr in tracks], False, (0, 255, 0))

            prev_subt_gray = frame_subt_gray

        prev_gray = frame_gray

        if frame_idx % detect_interval == 1:  # to find feature pts of a subtracted image, two images are required
            #mask = np.ones_like(frame_gray)
            mask = np.zeros_like(frame_gray)
            p = cv2.goodFeaturesToTrack(prev_subt_gray, mask=mask, **feature_params)
            if p is not None:
                for x, y in p.reshape(-1, 2):
                    tracks.append([(x, y)])

        if frame_idx > 1:
            vis = np.hstack((frame, vis))
            vis = imutils.resize(vis, width=600)
            cv2.imshow('result', vis)
            # cv2.imshow('result', frame_gray)
        frame_idx += 1
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            print("User interrupt!")
            break

        t_end = time.time()
        print("Elapsed time:", "%.3f" % round(t_end-t_start, 3), "sec\n")
        totalTime = totalTime + (t_end - t_start)

    print("Avg. processing time per frame =", round(totalTime / frame_idx, 3), "sec\n")
    vid.release()


#opticalflow_subtractedimgs()
opticalflow_subtractedimgs(args["video"])
#draw_of_subtractedimgs()
#draw_of_subtractedimgs(args["video"])
