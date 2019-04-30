import cv2
import numpy as np
import math
import time
from sklearn.metrics import pairwise

def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return tuple(nodes[np.argmin(dist_2)])

class BackgroundSubtract:
    def __init__(self, bgframe):
        self.aWeight = 0.5
        self.bg = bgframe

    def foreground(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        bg =  gray * self.aWeight + self.bg * (1 - self.aWeight)
        diff = cv2.absdiff(bg.astype(np.uint8),gray)
        _, diff = cv2.threshold(diff, 23,255,cv2.THRESH_BINARY)
        diff = np.repeat(diff[:, :, np.newaxis], 3, axis=2)
        return frame * diff * diff

class HandSegment:
    def calibrate(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        x, y, _ = frame.shape
        _, r, b = frame[int(x/2), int(y/2)]
        self.rmin = r - 10
        self.rmax = r + 12
        self.bmin = b - 10
        self.bmax = b + 12

    def extract(self, frame):
        minrange = np.array([0, self.rmin, self.bmin])
        maxrange = np.array([255, self.rmax, self.bmax])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        mask = cv2.inRange(frame, minrange, maxrange)
        return mask

    def denoise(self, frame):
        kernel = np.ones((15,15),np.uint8)
        erosion = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        reblur = cv2.blur(erosion,(35,35))
        _, final = cv2.threshold(reblur, 100, 255, cv2.THRESH_BINARY)
        return final

    def contour(self, frame):
        frame2 = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
        (contours, _) = cv2.findContours(frame,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return
        else:
            # find contours, convex hull, convexity defects, and bounding rectangle (centre of it too)
            segmented = max(contours, key=cv2.contourArea)
            chull = cv2.convexHull(segmented)
            defects = cv2.convexityDefects(segmented, cv2.convexHull(segmented, returnPoints=False))
            x,y,w,h = cv2.boundingRect(segmented)
            cX = int(x+w/2)
            cY = int(y+h/2)

            # create an ellipse around the contours (overlap over each finger) to find ROI
            roi = np.zeros(frame.shape[:2], dtype="uint8")
            ellipse = cv2.fitEllipse(segmented)
            cv2.ellipse(roi,ellipse,255,1)
            roi = cv2.bitwise_and(frame, frame, mask=roi)
            (cnts, _) = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # initialize final veriables, create an array of defect coorodinates
            count = 0
            results = []
            defectlist = np.array(list(segmented[defects[i,0][0]][0] for i in range(defects.shape[0])))

            # for each contour in the ROI, find the bounding rectangle
            # if the contour is not too far below in the wrist area, it is counted as a finger
            # closest defect to the bounding rectangle is the final point
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                midpoint = (int(x+w/2), int(y))
                if ((cY * 1.4) > (y + h)) and w*h > 30:
                    #cv2.rectangle(frame2, (x,y,w,h), (255,255,0), 1)
                    #cv2.circle(frame2, midpoint, 4, (255,0,0), -1)
                    results.append(closest_node(midpoint, defectlist))
                    #cv2.circle(frame2, finalpoint, 4, (0,255,0), -1)
                    count += 1
            #cv2.imshow("ayy", frame2)
            #cv2.waitKey(0)
            return results, count
bg = cv2.imread("images/0.JPG", 0)
framergb = cv2.imread("images/10.JPG")
bg = cv2.resize(bg, (0,0), fx=0.2, fy = 0.2)
framergb = cv2.resize(framergb, (0,0), fx=0.2, fy = 0.2)
#one = time.time()
#b = framergb
a = BackgroundSubtract(bg)
b = a.foreground(framergb)
#two = time.time()
c = HandSegment()
c.calibrate(framergb)
d = c.extract(b)
d = c.denoise(d)
#three = time.time()
#e = np.repeat(d[:, :, np.newaxis], 3, axis=2)
#cv2.imshow("hi",e*e*framergb)
#cv2.waitKey(0)
e = c.contour(d)
#four = time.time()
print(two-one, three-two, four-three, four-one)
print(e)
