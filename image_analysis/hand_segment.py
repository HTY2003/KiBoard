import cv2
import numpy as np

from image_helpers import graytorgb, closest_node, angle

class BackgroundSubtract:
    def __init__(self):
        self.aWeight = 0.5
        self.bg = None

    def calibrate(self, bgframe):
        if self.bg == None:
            self.bg = bgframe
        else:
            cv2.accumulateWeighted(bgframe, self.bg, self.aWeight)

    def foreground(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        bg =  gray * self.aWeight + self.bg * (1 - self.aWeight)
        diff = cv2.absdiff(bg.astype(np.uint8),gray)
        _, diff = cv2.threshold(diff, 15,255,cv2.THRESH_BINARY)
        return diff

class HandSegment:
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
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
        mask1 = cv2.inRange(ycrcb, minrange, maxrange)
        return mask1

    def denoise(self, frame):
        kernel = np.ones((9,9),np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        #frame = cv2.blur(frame,(17,30))
        frame = cv2.blur(frame,(8,10))
        _, final = cv2.threshold(frame, 20, 255, cv2.THRESH_BINARY)
        return final

    def contour(self, frame, display=None):
        frame2 = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
        if display.any():
            frame2 = frame2 * frame2 * display

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
            if display.any():
                cv2.ellipse(frame2,ellipse,(0,100,255),2)
            roi = cv2.bitwise_and(frame, frame, mask=roi)
            (cnts, _) = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # initialize final veriables, create an array of defect coorodinates
            count = 0
            results = []
            defectlist = []
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = segmented[s][0]
                end = segmented[e][0]
                far = segmented[f][0]
                degree = angle(start, far, end)
                if degree > 20 and degree < 110:
                    defectlist.append(end)
                    defectlist.append(start)
            defectlist = np.array(defectlist)
            ellipse = cv2.ellipse(np.zeros(frame.shape[:2], dtype="uint8"), ellipse, 255, -1)
            # for each contour in the ROI, find the bounding rectangle
            # if the contour is not too far below in the wrist area, it is counted as a finger
            # closest defect to the bounding rectangle is the final point
            for c in cnts:
                (x, y, w, h) = cv2.boundingRect(c)
                midpoint = (int(x+w/2), int(y-h*0.2))
                if ((cY * 1.5) > (y + h)) and w*h > 50:
                    finalpoint = closest_node(midpoint, defectlist, y+h*0.5)
                    if ellipse[finalpoint[1], finalpoint[0]] == 0:
                        results.append(finalpoint)
                        count += 1
                        if display.any():
                            cv2.rectangle(frame2, (x,y,w,h), (255,255,0), 1)
                            cv2.circle(frame2, midpoint, 3, (255,0,0), -1)
                            cv2.circle(frame2, finalpoint, 5, (0,255,0), -1)
            if display.any():
                cv2.imshow("ayy", frame2)
                cv2.waitKey(0)
            return results, count

class ShadowAnalysis:
    def __init__(self):
        self.above = 20
        self.below = 10
        self.left = 10
        self.right = 10

    def get_arrays(self, frame, coordinates):
        array_list = []
        for x,y in coordinates:
            left = x - self.left
            right = x + self.right
            top = y - self.top
            bottom = y + self.bottom
            array_list.append(frame[left:right, top:bottom, :])
        return array_list

    def extract(self, array_list):
        percentage_list = []
        minrange = np.array([0, 0, 0])
        maxrange = np.array([255, int(0.4*255), 255])
        for array in array_list:
            hls = cv2.cvtColor(rect, cv2.COLOR_BGR2HLS)
            mask = cv2.inRange(hls, minrange, maxrange)
            size = rect.shape[0] * rect.shape[1]
            percentage = cv2.countNonZero(cv2.threshold(mask,0,255,cv2.THRESH_BINARY))/size
            percentage_list.append(percentage)
        return percentage_list

'''bg = cv2.imread("images/0.JPG", 0)
framergb = cv2.imread("images/4.JPG")
bg = cv2.resize(bg, (0,0), fx=0.2, fy = 0.2)
framergb = cv2.resize(framergb, (0,0), fx=0.2, fy = 0.2)

a = BackgroundSubtract(bg)
b = a.foreground(framergb)
c = graytorgb(b, framergb)
#cv2.imshow("hi",b)
#cv2.waitKey(0)
d = HandSegment()
d.calibrate(framergb)
e = d.extract(c)
i = d.denoise(e)
h = d.contour(i, framergb)
print(h)

f = ShadowSegment()
z,g = f.extract(framergb)
print(g)
z = graytorgb(z, framergb)
cv2.imshow("gay shut",z)
cv2.waitKey(0)'''