import sys
import cv2
import time
from cvzone.PoseModule import PoseDetector
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from filter import *



# source image
imgPath = './Image/template/'
img_A = cv2.imread(imgPath + 'A.png', 3)
img_B = cv2.imread(imgPath + 'B.png', 3)
img_out = cv2.imread(imgPath + 'A.png', 3)
H = img_A.shape[0]
W = img_A.shape[1]
if img_B.shape[0] != H or img_B.shape[1] != W:
    sys.exit(" QAQ Error: two images have different sizes")

# temptlate (offset, size) (y,x)
tplt = np.array([[600, 420, 472, 2530],  # block0 : Head
                 [1110, 420, 2340, 4100],  # block1 : Pic
                 [3480, 420, 820, 4520],  # block2 : Paragraph
                 [140, 5300, 800, 2500],  # block3 : Weather
                 [1240, 5300, 1250, 2490],  # block4 : Edit fav
                 [2890, 5300, 1500, 2490]])  # block5 : Comment

# resizing
reduceFactor = 8
img_A = cv2.resize(img_A, None, fx=(1 / reduceFactor), fy=(1 / reduceFactor))
img_B = cv2.resize(img_B, None, fx=(1 / reduceFactor), fy=(1 / reduceFactor))
img_out = cv2.resize(img_out, None, fx=(1 / reduceFactor), fy=(1 / reduceFactor))
tplt = tplt // reduceFactor


# source segment
img_segment_0A = img_A[tplt[0, 0]:tplt[0, 0] + tplt[0, 2], tplt[0, 1]:tplt[0, 1] + tplt[0, 3], :]
img_segment_0B = img_B[tplt[0, 0]:tplt[0, 0] + tplt[0, 2], tplt[0, 1]:tplt[0, 1] + tplt[0, 3], :]
img_segment_1A = img_A[tplt[1, 0]:tplt[1, 0] + tplt[1, 2], tplt[1, 1]:tplt[1, 1] + tplt[1, 3], :]
img_segment_1B = img_B[tplt[1, 0]:tplt[1, 0] + tplt[1, 2], tplt[1, 1]:tplt[1, 1] + tplt[1, 3], :]
img_segment_2A = img_A[tplt[2, 0]:tplt[2, 0] + tplt[2, 2], tplt[2, 1]:tplt[2, 1] + tplt[2, 3], :]
img_segment_2B = img_B[tplt[2, 0]:tplt[2, 0] + tplt[2, 2], tplt[2, 1]:tplt[2, 1] + tplt[2, 3], :]
img_segment_3A = img_A[tplt[3, 0]:tplt[3, 0] + tplt[3, 2], tplt[3, 1]:tplt[3, 1] + tplt[3, 3], :]
img_segment_3B = img_B[tplt[3, 0]:tplt[3, 0] + tplt[3, 2], tplt[3, 1]:tplt[3, 1] + tplt[3, 3], :]
img_segment_4A = img_A[tplt[4, 0]:tplt[4, 0] + tplt[4, 2], tplt[4, 1]:tplt[4, 1] + tplt[4, 3], :]
img_segment_4B = img_B[tplt[4, 0]:tplt[4, 0] + tplt[4, 2], tplt[4, 1]:tplt[4, 1] + tplt[4, 3], :]
img_segment_5A = img_A[tplt[5, 0]:tplt[5, 0] + tplt[5, 2], tplt[5, 1]:tplt[5, 1] + tplt[5, 3], :]
img_segment_5B = img_B[tplt[5, 0]:tplt[5, 0] + tplt[5, 2], tplt[5, 1]:tplt[5, 1] + tplt[5, 3], :]

# status variable & parameter
blockTrigger = [False, False, False, False, False, False]
# effectTrigger = [False, False, False, False, False, False]
effectTrigger = [True, True, True, True, True, True]

# detector & capture video
detector = PoseDetector()
cap = cv2.VideoCapture('output_1_Compressed.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
# print("cap.isOpened:", cap.isOpened())
TIMER = int(60)
prev = time.time()
while True:
    while TIMER >= 0:
        cur = time.time()

        success, frame = cap.read()
        if not success:
            print("cap.read failed")
            break
        frame = cv2.resize(frame, None, fx=(1 / 4), fy=(1 / 4))
        cap_width = frame.shape[1]
        cap_height = frame.shape[0]
        detector.findPose(frame, draw=False)
        lmlist, bbox = detector.findPosition(frame, draw=False)

        if bbox:
            center = bbox["center"]
            # 0
            if 0 < center[0] < cap_width // 3 and 0 < center[1] < cap_height // 2:
                cv2.putText(frame, "Block 0", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 150, 255), 2)
                blockTrigger[0] = True
            elif cap_width // 3 < center[0] or center[1] > cap_height // 2:
                blockTrigger[0] = False
                cv2.imshow('News', img_out)
                cv2.waitKey(1)
            # 1
            if cap_width // 3 < center[0] < (cap_width // 3) * 2 and center[1] < cap_height // 2:
                cv2.putText(frame, "Block 1", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 150, 255), 2)
                blockTrigger[1] = True
                img_out[tplt[1, 0]:tplt[1, 0] + tplt[1, 2], tplt[1, 1]:tplt[1, 1] + tplt[1, 3], :] = img_segment_1B
            elif center[0] < cap_width // 3 or center[0] > (cap_width // 3) * 2 or center[1] > cap_height // 2:
                blockTrigger[1] = False
                img_out[tplt[1, 0]:tplt[1, 0] + tplt[1, 2], tplt[1, 1]:tplt[1, 1] + tplt[1, 3], :] = img_segment_1A

            # 2
            if (cap_width // 3) * 2 < center[0] < cap_width and center[1] < cap_height // 2:
                cv2.putText(frame, "Block 2", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 150, 255), 2)
                blockTrigger[2] = True
                img_out[tplt[2, 0]:tplt[2, 0] + tplt[2, 2], tplt[2, 1]:tplt[2, 1] + tplt[2, 3], :] = img_segment_2B
            elif center[0] < (cap_width // 3) * 2 or center[1] > cap_height // 2:
                blockTrigger[2] = False
                img_out[tplt[2, 0]:tplt[2, 0] + tplt[2, 2], tplt[2, 1]:tplt[2, 1] + tplt[2, 3], :] = img_segment_2A

            # 3
            if 0 < center[0] < cap_width // 3 and cap_height // 2 < center[1]:
                cv2.putText(frame, "Block 3", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 150, 255), 2)
                blockTrigger[3] = True
                img_out[tplt[3, 0]:tplt[3, 0] + tplt[3, 2], tplt[3, 1]:tplt[3, 1] + tplt[3, 3], :] = img_segment_3B
            elif center[0] > cap_width // 3 or cap_height // 2 > center[1]:
                blockTrigger[3] = False
                img_out[tplt[3, 0]:tplt[3, 0] + tplt[3, 2], tplt[3, 1]:tplt[3, 1] + tplt[3, 3], :] = img_segment_3A

            # 4
            if cap_width // 3 < center[0] < (cap_width // 3) * 2 and cap_height // 2 < center[1] < cap_height:
                cv2.putText(frame, "Block 4", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 150, 255), 2)
                blockTrigger[4] = True
                img_out[tplt[4, 0]:tplt[4, 0] + tplt[4, 2], tplt[4, 1]:tplt[4, 1] + tplt[4, 3], :] = img_segment_4B
            elif center[0] < cap_width // 3 or center[0] > (cap_width // 3) * 2 or cap_height // 2 > center[1]:
                blockTrigger[4] = False
                img_out[tplt[4, 0]:tplt[4, 0] + tplt[4, 2], tplt[4, 1]:tplt[4, 1] + tplt[4, 3], :] = img_segment_4A

            # 5
            if (cap_width // 3) * 2 < center[0] < cap_width and cap_height // 2 < center[1] < cap_height:
                cv2.putText(frame, "Block 5", (100, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 150, 255), 2)
                blockTrigger[5] = True
                img_out[tplt[5, 0]:tplt[5, 0] + tplt[5, 2], tplt[5, 1]:tplt[5, 1] + tplt[5, 3], :] = img_segment_5B
            elif (cap_width // 3) * 2 > center[0] or cap_height // 2 > center[1]:
                blockTrigger[5] = False
                img_out[tplt[5, 0]:tplt[5, 0] + tplt[5, 2], tplt[5, 1]:tplt[5, 1] + tplt[5, 3], :] = img_segment_5A

            cv2.putText(frame, str(center), (100, 500), cv2.FONT_HERSHEY_PLAIN, 2, (0, 150, 255), 2)
            cv2.putText(frame, str(blockTrigger), (100, 300), cv2.FONT_HERSHEY_PLAIN, 2, (0, 150, 255), 2)
            cv2.circle(frame, center, 5, (100, 255, 100), -1)

        # effect trigger detection
        # block0 fade filter
        if blockTrigger[0] and effectTrigger[0]:
            for i in np.linspace(0, 1, 100):
                alpha = i
                beta = 1 - alpha
                output = cv2.addWeighted(img_segment_0B, alpha, img_segment_0A, beta, 0)
                img_out[tplt[0, 0]:tplt[0, 0] + tplt[0, 2], tplt[0, 1]:tplt[0, 1] + tplt[0, 3], :] = output
                cv2.imshow('News', img_out)
                cv2.waitKey(1)
                effectTrigger[0] = False
        if not blockTrigger[0] and not effectTrigger[0]:
            for i in np.linspace(0, 1, 100):
                alpha = i
                beta = 1 - alpha
                output = cv2.addWeighted(img_segment_0A, alpha, img_segment_0B, beta, 0)
                img_out[tplt[0, 0]:tplt[0, 0] + tplt[0, 2], tplt[0, 1]:tplt[0, 1] + tplt[0, 3], :] = output
                cv2.imshow('News', img_out)
                cv2.waitKey(1)
                effectTrigger[0] = True

        # block1 morph filter
        if blockTrigger[1] and effectTrigger[1]:
            pts1 = np.array([[218, 240], [295, 240], [250, 383]], np.float32)
            pts2 = np.array([[248, 245], [345, 270], [281, 366]], np.float32)
            pts11 = np.zeros((3, 2), np.float32)
            pts22 = np.zeros((3, 2), np.float32)
            dis = 100.0  # iterations
            piece = 1.0 / dis
            for i in range(0, int(dis)):
                for j in range(0, 3):
                    disx = (pts1[j, 0] - pts2[j, 0]) * -1
                    disy = (pts1[j, 1] - pts2[j, 1]) * -1
                    # move of first image
                    movex1 = (disx / dis) * (i + 1)
                    movey1 = (disy / dis) * (i + 1)
                    # move of second image
                    movex2 = disx - movex1
                    movey2 = disy - movey1
                    pts11[j, 0] = pts1[j, 0] + movex1
                    pts11[j, 1] = pts1[j, 1] + movey1
                    pts22[j, 0] = pts2[j, 0] - movex2
                    pts22[j, 1] = pts2[j, 1] - movey2
                mat1 = cv2.getAffineTransform(pts1, pts11)
                mat2 = cv2.getAffineTransform(pts2, pts22)
                dst1 = cv2.warpAffine(img_segment_1A, mat1, (img_segment_1A.shape[1], img_segment_1A.shape[0]), None,
                                      None, cv2.BORDER_REPLICATE)
                dst2 = cv2.warpAffine(img_segment_1B, mat2, (img_segment_1B.shape[1], img_segment_1B.shape[0]), None,
                                      None, cv2.BORDER_REPLICATE)
                dst = cv2.addWeighted(dst1, 1 - (piece * (i)), dst2, piece * (i + 1), 0)
                img_out[tplt[1, 0]:tplt[1, 0] + tplt[1, 2], tplt[1, 1]:tplt[1, 1] + tplt[1, 3], :] = dst
                cv2.imshow("News", img_out)
                cv2.waitKey(1)
                effectTrigger[1] = False
        if not blockTrigger[1] and not effectTrigger[1]:
            pts1 = np.array([[218, 240], [295, 240], [250, 383]], np.float32)
            pts2 = np.array([[248, 245], [345, 270], [281, 366]], np.float32)
            pts11 = np.zeros((3, 2), np.float32)
            pts22 = np.zeros((3, 2), np.float32)
            dis = 100.0  # iterations
            piece = 1.0 / dis
            for i in range(0, int(dis)):
                for j in range(0, 3):
                    disx = (pts1[j, 0] - pts2[j, 0]) * -1
                    disy = (pts1[j, 1] - pts2[j, 1]) * -1
                    # move of first image
                    movex1 = (disx / dis) * (i + 1)
                    movey1 = (disy / dis) * (i + 1)
                    # move of second image
                    movex2 = disx - movex1
                    movey2 = disy - movey1
                    pts11[j, 0] = pts1[j, 0] + movex1
                    pts11[j, 1] = pts1[j, 1] + movey1
                    pts22[j, 0] = pts2[j, 0] - movex2
                    pts22[j, 1] = pts2[j, 1] - movey2
                mat1 = cv2.getAffineTransform(pts1, pts11)
                mat2 = cv2.getAffineTransform(pts2, pts22)
                dst1 = cv2.warpAffine(img_segment_1B, mat1, (img_segment_1B.shape[1], img_segment_1B.shape[0]), None,
                                      None, cv2.BORDER_REPLICATE)
                dst2 = cv2.warpAffine(img_segment_1A, mat2, (img_segment_1A.shape[1], img_segment_1A.shape[0]), None,
                                      None, cv2.BORDER_REPLICATE)
                dst = cv2.addWeighted(dst1, 1 - (piece * (i)), dst2, piece * (i + 1), 0)
                img_out[tplt[1, 0]:tplt[1, 0] + tplt[1, 2], tplt[1, 1]:tplt[1, 1] + tplt[1, 3], :] = dst
                cv2.imshow("News", img_out)
                cv2.waitKey(1)
                effectTrigger[1] = True

        # block 2 running filter
        if blockTrigger[2] and effectTrigger[2]:
            h, w, c = img_A.shape
            for k in range(0, 600):
                l = img_segment_2B[:, :(k % w)]
                r = img_segment_2B[:, (k % w):]
                output2 = np.hstack((r, l))
                img_out[tplt[2, 0]:tplt[2, 0] + tplt[2, 2], tplt[2, 1]:tplt[2, 1] + tplt[2, 3], :] = output2
                cv2.imshow("News", img_out)
                cv2.waitKey(1)
                effectTrigger[2] = False
        if not blockTrigger[2] and not effectTrigger[2]:
            h, w, c = img_A.shape
            for k in range(0, 600):
                l = img_segment_2A[:, :(k % w)]
                r = img_segment_2A[:, (k % w):]
                output2 = np.hstack((r, l))
                img_out[tplt[2, 0]:tplt[2, 0] + tplt[2, 2], tplt[2, 1]:tplt[2, 1] + tplt[2, 3], :] = output2
                cv2.imshow("News", img_out)
                cv2.waitKey(1)
                effectTrigger[2] = True

        # block 3 smoothing filter
        if blockTrigger[3] and effectTrigger[3]:
            for i in np.linspace(0, 1, 100):
                alpha = i
                beta = 1 - alpha
                output3 = cv2.addWeighted(img_segment_3B, alpha, glow(img_segment_3A), beta, 0)
                img_out[tplt[3, 0]:tplt[3, 0] + tplt[3, 2], tplt[3, 1]:tplt[3, 1] + tplt[3, 3], :] = output3
                cv2.imshow("News", img_out)
                cv2.waitKey(1)
                effectTrigger[3] = False
        if not blockTrigger[3] and not effectTrigger[3]:
            for i in np.linspace(0, 1, 100):
                alpha = i
                beta = 1 - alpha
                output3 = cv2.addWeighted(glow(img_segment_3A), alpha, img_segment_3B, beta, 0)
                img_out[tplt[3, 0]:tplt[3, 0] + tplt[3, 2], tplt[3, 1]:tplt[3, 1] + tplt[3, 3], :] = output3
                cv2.imshow("News", img_out)
                cv2.waitKey(1)
                effectTrigger[3] = True

        # block 4 smoothing filter
        if blockTrigger[4] and effectTrigger[4]:
            for i in np.linspace(0, 1, 100):
                alpha = i
                beta = 1 - alpha
                output4 = cv2.addWeighted(img_segment_4B, alpha, smoothing(img_segment_4A), beta, 0)
                img_out[tplt[4, 0]:tplt[4, 0] + tplt[4, 2], tplt[4, 1]:tplt[4, 1] + tplt[4, 3], :] = output4
                cv2.imshow("News", img_out)
                cv2.waitKey(1)
                effectTrigger[4] = False
        if not blockTrigger[4] and not effectTrigger[4]:
            for i in np.linspace(0, 1, 100):
                alpha = i
                beta = 1 - alpha
                output4 = cv2.addWeighted(smoothing(img_segment_4A), alpha, img_segment_4B, beta, 0)
                img_out[tplt[4, 0]:tplt[4, 0] + tplt[4, 2], tplt[4, 1]:tplt[4, 1] + tplt[4, 3], :] = output4
                cv2.imshow("News", img_out)
                cv2.waitKey(1)
                effectTrigger[4] = True

        # block 5 smoothing filter
        if blockTrigger[5] and effectTrigger[5]:
            for i in np.linspace(0, 1, 100):
                alpha = i
                beta = 1 - alpha
                output5 = cv2.addWeighted(img_segment_5B, alpha, binary(img_segment_5A), beta, 0)
                img_out[tplt[5, 0]:tplt[5, 0] + tplt[5, 2], tplt[5, 1]:tplt[5, 1] + tplt[5, 3], :] = output5
                cv2.imshow("News", img_out)
                cv2.waitKey(1)
                effectTrigger[5] = False
        if not blockTrigger[5] and not effectTrigger[5]:
            for i in np.linspace(0, 1, 100):
                alpha = i
                beta = 1 - alpha
                output5 = cv2.addWeighted(binary(img_segment_5A), alpha, img_segment_5B, beta, 0)
                img_out[tplt[5, 0]:tplt[5, 0] + tplt[5, 2], tplt[5, 1]:tplt[5, 1] + tplt[5, 3], :] = output5
                cv2.imshow("News", img_out)
                cv2.waitKey(1)
                effectTrigger[5] = True
        cv2.line(frame, (0, cap_height//2), (cap_width, cap_height//2), (0, 0, 255), 3)
        cv2.line(frame, (cap_width // 3 * 2, 0), (cap_width // 3 * 2, cap_height), (0, 0, 255), 3)
        cv2.line(frame, (cap_width // 3, 0), (cap_width // 3, cap_height), (0, 0, 255), 3)
        cv2.putText(frame, str(TIMER), (400, 250), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.imshow("News", img_out)
        cv2.imshow("cam", frame)

        if cur - prev >= 1:
            prev = cur
            TIMER = TIMER - 1

            print(TIMER)
        if TIMER == -1:
            cv2.destroyAllWindows()
        cv2.waitKey(1)


