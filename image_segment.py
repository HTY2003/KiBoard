import cv2
import imutils
import numpy as np
import math
from sklearn.metrics import pairwise
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

class ColorSegment:
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
    
    def calculateFingers(self, res,drawing):  # -> finished bool, cnt: finger count
        #  convexity defect
        hull = cv2.convexHull(res, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(res, hull)
            if type(defects) != type(None):  # avoid crashing.   (BUG not found)

                cnt = 0
                for i in range(defects.shape[0]):  # calculate the angle
                    s, e, f, d = defects[i][0]
                    start = tuple(res[s][0])
                    end = tuple(res[e][0])
                    far = tuple(res[f][0])
                    a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                    c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                    if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                        cnt += 1
                        cv2.circle(drawing, far, 8, [211, 84, 0], -1)
                return True, cnt
        return False, 0

    def contour(self, frame):
        frame2 = np.repeat(frame[:, :, np.newaxis], 3, axis=2)
        (contours, _) = cv2.findContours(frame,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        if length == 0:
            return
        else:
            maxArea = -1
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(frame.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

            isFinishCal,cnt = self.calculateFingers(res,drawing)
            '''cnts = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(cnts)
            hull2 = cv2.convexHull(cnts,returnPoints = False)
            defects = cv2.convexityDefects(cnts,hull2)
            
            FarDefect = []
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(cnts[s][0])
                end = tuple(cnts[e][0])
                far = tuple(cnts[f][0])
                FarDefect.append(far)
                cv2.line(frame,start,end,[0,255,0],1)
                cv2.circle(frame,far,10,[100,255,255],3)
            
            moments = cv2.moments(cnts)
            
            if moments['m00']!=0:
                cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                cy = int(moments['m01']/moments['m00']) # cy = M01/M00
            centerMass=(cx,cy)    
            
            cv2.circle(frame,centerMass,7,[100,0,255],2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,'Center',tuple(centerMass),font,2,(255,255,255),2)     
            
            distanceBetweenDefectsToCenter = []
            for i in range(0,len(FarDefect)):
                x =  np.array(FarDefect[i])
                centerMass = np.array(centerMass)
                distance = np.sqrt(np.power(x[0]-centerMass[0],2)+np.power(x[1]-centerMass[1],2))
                distanceBetweenDefectsToCenter.append(distance)
            
            sortedDefectsDistances = sorted(distanceBetweenDefectsToCenter)
            AverageDefectDistance = np.mean(sortedDefectsDistances[0:2])
        
            finger = []
            for i in range(0,len(hull)-1):
                if (np.absolute(hull[i][0][0] - hull[i+1][0][0]) > 80) or ( np.absolute(hull[i][0][1] - hull[i+1][0][1]) > 80):
                    if hull[i][0][1] <500 :
                        finger.append(hull[i][0])
            
            finger =  sorted(finger,key=lambda x: x[1])   
            fingers = finger[0:5]
            
            fingerDistance = []
            for i in range(0,len(fingers)):
                distance = np.sqrt(np.power(fingers[i][0]-centerMass[0],2)+np.power(fingers[i][1]-centerMass[0],2))
                fingerDistance.append(distance)
            
            result = 0
            for i in range(0,len(fingers)):
                if fingerDistance[i] > AverageDefectDistance+130:
                    result += 1'''
            '''segmented = max(contours, key=cv2.contourArea)
            chull2 = cv2.convexHull(segmented, returnPoints=False)
            defects = cv2.convexityDefects(segmented,chull2)
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(segmented[s][0])
                end = tuple(segmented[e][0])
                far = tuple(segmented[f][0])
                cv2.line(frame2,start,end,[0,255,0],5)
                cv2.circle(frame2,far,5,[0,0,255],-1)
            cv2.imshow('img',frame2)
            cv2.waitKey(0)'''
            
            return cnt

bg = cv2.imread("images/0.JPG", 0)
framergb = cv2.imread("images/4.JPG")
bg = cv2.resize(bg, (0,0), fx=0.2, fy = 0.2)
framergb = cv2.resize(framergb, (0,0), fx=0.2, fy = 0.2)
a = BackgroundSubtract(bg)
b = a.foreground(framergb)
c = ColorSegment()
c.calibrate(framergb)
d = c.extract(b)
d = c.denoise(d)
e = np.repeat(d[:, :, np.newaxis], 3, axis=2)
cv2.imshow("hi",e*e*framergb)
cv2.waitKey(0)
print(c.contour(d))
