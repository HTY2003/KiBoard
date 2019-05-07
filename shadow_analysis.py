import cv2
import numpy as np
import math
import time

from image_helpers import graytorgb

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
