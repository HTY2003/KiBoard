import cv2
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing

np.set_printoptions(threshold=np.inf)

def blockify(arr, nrows, ncols):
   h, w = arr.shape
   return (arr.reshape(h//nrows, nrows, -1, ncols)
           .swapaxes(1,2)
           .reshape(-1, nrows, ncols))

def crop(img, cell_size):
    h, w = img.shape
    #crop height and width to be divisble by 8
    excessh = h - h % cell_size
    excessw = w - w % cell_size
    img = img[:excessh, :excessw]
    return img

def SobelX(img):
    return cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
def SobelY(img):
    return cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

def process(img):
    #run img through Sobel operators (edge detection)
    gx, gy = SobelX(img), SobelY(img)
    #gx, gy = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5), cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    # calculate magnitude and direction (in angles) of the image
    # magnitude is calculated using  (dx**2 + dy**2) // 2
    # gradient is calculated using arctan(dy / dx)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag, angle = np.absolute(mag), np.absolute(angle)
    return mag, angle

def unsigned_angles(cell):
    return np.add(np.multiply(cell[cell > 180], -1), 360, out=cell[cell > 180])

def angle_hog(angle, mag, cell_size):
    #filter angles into a 0-180deg 9-bin histogram
    histolist = []
    acell = blockify(angle, 8, 8)
    mcell = blockify(mag, 8, 8)
    for i in range(len(acell)):
        cell = acell[i]
        #make the gradients unsigned (negative angles < 180 = positive)
        cell = np.uint8(unsigned_angles(cell))

        #split the magnitude between 2 angles, add each value to its slot in the histogram
        histogram = np.zeros(9, dtype=float)
        for x in range(cell_size):
            for x2 in range(cell_size):
                a = acell[i][x][x2]
                base = int((a-(a%20)) % 180 / 20)
                base2 = (base+1) % 9
                ratio = float((a%20) / 20)
                ratio2 = 1 - ratio
                histogram[base] += mcell[i][x][x2] * ratio
                histogram[base2] += mcell[i][x][x2] * ratio2
        histolist.append(histogram)
    #add 9-bin histogram to list
    #histolist.append(np.histogram(cell, bins=9, range=(0,160))[0])
    #convert list into nested lists, then 3-dimensional numpy matrix (y by x by 9)
    h, w = angle.shape
    n1, n2 = int(h / cell_size), int(w / cell_size)
    histolist = [histolist[i*n2:i*n2+n2] for i in range(0, n1)]
    histolist = np.array(histolist)
    return histolist

def mag_normalize(mag):
    return

def show(img, mag):
    plt.subplot(1,2,1),plt.imshow(img,cmap = 'viridis')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(mag,cmap = 'viridis')
    plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':
    img = cv2.imread('images/download.jpg',0)
    img = crop(img, 8)
    mag, angle = process(img)
    result = angle_hog(angle, mag, 8)
    #show(img, mag)
