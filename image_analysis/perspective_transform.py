class Perspective_Transform:
    def __init__(self):
        self.points = np.zeros((4, 2), dtype = "float32")

    def findPoints(self, frame):
        points = []
        wow = []
        minrange = np.array([200,0,200])
        maxrange = np.array([255,40,255])
        mask = cv2.inRange(frame, minrange, maxrange)

        im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append([cX, cY])

        if len(self.points) == 4:
            self.points = self.sortPoints(np.array(points))

    def sortPoints(points):
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def transform(frame,x,y):
    	(tl, tr, br, bl) = self.points
        frame = np.zeros(frame.shape)
        frame[y,x] = 255

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    	maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    	maxHeight = max(int(heightA), int(heightB))
        dst = np.array([
    		[0, 0],
    		[maxWidth - 1, 0],
    		[maxWidth - 1, maxHeight - 1],
    		[0, maxHeight - 1]], dtype = "float32")
    	M = cv2.getPerspectiveTransform(rect, dst)
    	warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    	return warped
