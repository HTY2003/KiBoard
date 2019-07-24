import cv2
import time
from image_analysis.colour_segment import ColourSegment
from image_analysis.image_helpers import *
from async_capture import VideoCaptureAsync

vc = VideoCaptureAsync(0)
a = ColourSegment()

vc.start()

#if vc.isOpened():
#    rval, frame = vc.read()
#else:
#    rval = False
rval = False
print("Please put your hand in the centre")
for i in range(100):
    rval, frame = vc.read()
    frame = cv2.flip(frame,1)
    frame = draw_rect(frame)
    cv2.imshow("preview", frame)
    key = cv2.waitKey(20)
    if key == 27:
        break

rval, frame = vc.read()
frame = cv2.flip(frame,1)
hist = hand_histogram(frame)
counter = 0
history = []

while rval:
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    frame = cv2.flip(frame,1)
    frame1 = hist_masking(frame, hist)
    frame2 = block_face(frame, frame1)
    frame3 = denoise(frame, frame2)
    results, count, frame4 = a.contour(frame3, frame)
    if count == 5:
        counter = 100

    if count == 0:
        counter = 200

    if count == 2 or count == 3:
        results = sorted(results, key=lambda k:[k[1],k[0]])
        history.append(results[0])

    for i in history:
        cv2.circle(frame4, i, 10, (0,0,255), -1)
    if frame4 is not None:
        cv2.imshow("preview2", frame4)
    key = cv2.waitKey(20)
    if key == 27:
        break
    if counter == 100:
        if len(history) > 20:
            with open("test.txt", "a") as f:
                f.write(str(history) + "\n")
                f.close()
        history = []
        counter = 0

    if counter == 200:
        history = []
        counter = 0
cv2.destroyWindow("preview")
vc.stop()
