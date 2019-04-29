import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import time, math

class BackgroundSubtract:
    def __init__(self, bgframe):
        self.aWeight = 0.5
        self.bg = bgframe

    def foreground(self, frame):
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        bg =  frame * self.aWeight + self.bg * (1 - self.aWeight)
        diff = cv2.absdiff(bg.astype(np.uint8),frame)

class ColorSegment:
    def calibrate(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        x, y, _ = frame.shape
        _, r, b = frame[int(x/2), int(y/2)]
        self.rmin = r - 10
        self.rmax = r + 10
        self.bmin = b - 10
        self.bmax = b + 10

    def extract(self, frame):
        minrange = np.array([0, self.rmin, self.bmin])
        maxrange = np.array([255, self.rmax, self.bmax])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        mask = cv2.inRange(frame, minrange, maxrange)
        return mask

    def denoise(self, frame):
        kernel = np.ones((20,20),np.uint8)
        blur = cv2.blur(frame,(5,5))
        erosion = cv2.morphologyEx(blur, cv2.MORPH_OPEN, kernel)
        return erosion

    def contour(self, frame):
        (contours, _) = cv2.findContours(frame,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return
        else:
            segmented = max(contours, key=cv2.contourArea)
            chull = cv2.convexHull(segmented)
            extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
            extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
            extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
            extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

            cX = int((extreme_left[0] + extreme_right[0]) / 2)
            cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
            distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
            maximum_distance = distance[distance.argmax()]
            
            radius = int(0.8 * maximum_distance)
            circumference = (2 * np.pi * radius)


            circular_roi = np.zeros(frame.shape[:2], dtype="uint8")
            print(cX, cY)
            cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
            
            circular_roi = cv2.bitwise_and(frame, frame, mask=circular_roi)
            (contours, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            count = 0

            for c in contours:
                (x, y, w, h) = cv2.boundingRect(c)
                count += 1
            return count

framergb = cv2.imread("images/6.JPG")
c = time.time()
framergb = cv2.resize(framergb, (0,0), fx=0.2, fy = 0.2)
a = ColorSegment()
a.calibrate(framergb)
b = a.extract(framergb)
b = a.denoise(b)
#cv2.imshow("hi",b)
#cv2.waitKey(0)
print(a.contour(b))