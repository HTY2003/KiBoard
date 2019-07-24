import cv2
import numpy as np
class BackgroundSubtract:
    def __init__(self):
        self.aWeight = 0.5
        self.bg = []

    def calibrate(self, bgframe):
        bgframe = cv2.blur(bgframe, (3,3))
        if len(self.bg) == 0:
            self.bg = bgframe
        else:
            self.bg = self.bg * 0.5 + bgframe * 0.5

    def foreground(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3,3))
        bg =  gray * self.aWeight + self.bg * (1 - self.aWeight)
        diff = cv2.absdiff(bg.astype(np.uint8),gray)
        kernel = np.ones((3,3),np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_ELLIPSE, kernel)
        diff = cv2.GaussianBlur(diff, (5, 5), 0)
        _, diff = cv2.threshold(diff, 5,255,cv2.THRESH_BINARY)
        return diff
