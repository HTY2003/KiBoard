import cv2

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
