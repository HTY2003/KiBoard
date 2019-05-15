import cv2
import numpy as np

from image_helpers import graytorgb, closest_node, angle

class ColourSegment:
    def extract(self, frame):
        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]
        t1 = np.where(r-b > 0, True, False)
        t2 = np.where(r-g > 15, True, False)
        mask1 = np.logical_and(t1,t2)

        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        u = yuv[:, :, 1]
        v = yuv[:, :, 2]
        sat = np.sqrt(np.square(u) + np.square(v))
        mask2 = np.where(sat <= 220, True, False)

        minrange = np.array([20,40,95])
        maxrange = np.array([255,255,255])
        mask3 = cv2.inRange(frame, minrange, maxrange)

        return np.logical_and(mask1,mask2) * mask3

    def shadow(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask1 = np.where(gray < 180, True, False)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        minrange = np.array([0,0,0])
        maxrange = np.array([255,int(0.2 * 255),185])
        mask2 = cv2.inRange(hsv, minrange, maxrange)

        return mask1 * mask2

    def denoise(self, frame):
        kernel = np.ones((5,5),np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        frame = cv2.blur(frame,(6,12))
        _, final = cv2.threshold(frame, 15, 255, cv2.THRESH_BINARY)
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
