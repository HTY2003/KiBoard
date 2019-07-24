import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('C:\\Users\\randu\\AppData\\Local\\Programs\\Python\\Python36\\Lib\\site-packages\cv2\data\haarcascade_frontalface_default.xml')

def block_face(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(mask,(x,y-int(h*0.4)),(x+int(w* 1.8),int(y+h*1.8)),0,-1)
    return mask

def graytorgb(mask, framergb):
    frame = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    return frame * frame * framergb

def closest_node(node, nodes, ymin):
    if len(nodes) == 0:
        return (0,0)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    closest = nodes[np.argmin(dist_2)]
    return tuple(closest)

def angle(start, centre, end):
    ba = start - centre
    bc = end - centre
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def draw_rect(frame):
    rows, cols, _ = frame.shape
    global hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)
    hand_rect_one_x -= 5
    hand_rect_one_y -= 5
    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(9):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)
    for i in range(9):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]
    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 5, 255, cv2.NORM_MINMAX)

def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 0.1)
    dst = dst * dst * dst * 255
    dst = cv2.blur(dst, (13,13))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY)
    #thresh = cv2.merge((thresh, thresh, thresh))
    #return cv2.bitwise_and(frame, thresh)
    return thresh

def denoise(frame, mask):
    kernel = np.ones((15,15),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.blur(mask, (3,3))
    _, mask = cv2.threshold(mask, 2, 255, cv2.THRESH_BINARY)
    final = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #final = cv2.merge((final, final, final))
    #return cv2.bitwise_and(frame, final)
    final = cv2.blur(mask, (5,5))
    _, final = cv2.threshold(final, 5, 255, cv2.THRESH_BINARY)
    return final
