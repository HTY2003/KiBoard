import timeit
import cv2
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing

np.set_printoptions(threshold=np.inf)

def blockify(arr, nrows, ncols):
   h, w = arr.shape[:2]
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
    return cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
def SobelY(img):
    return cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

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
    np.multiply(angle[angle > 180], -1, out=angle[angle > 180])
    np.add(angle[angle < -180], 360, out=angle[angle < -180])
    combi = np.vstack(([mag.T], [angle.T])).T
    combi = blockify(combi, cell_size, cell_size)
    len = combi.shape[0]
    combi = combi.reshape(len, cell_size**2)
    histolist = np.zeros((len, 9))
    print(combi[2])
    for i in range(len):
        for a, m in combi[i]:
            mint = int(m)
            #split the magnitude between 2 angles, add each value to its slot in the histogram
            histogram = np.zeros(9)
            base = int((a1-(a1%20)) % 180 / 20)
            base2 = (base+1) % 9
            ratio = float((a1%20) / 20)
            ratio2 = 1 - ratio
            histogram[base] += mint * ratio
            histogram[base2] += mint * ratio2
        histolist[i] = histogram
    #add 9-bin histogram to list
    #histolist.append(np.histogram(cell, bins=9, range=(0,160))[0])
    #convert list into nested lists, then 3-dimensional numpy matrix (y by x by 9)
    #print(3, np.uint16(combi))
    h, w = angle.shape
    n1, n2 = int(h / cell_size), int(w / cell_size)
    histolist = histolist.reshape(n1, n2, 9)
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
    setup = '''
import timeit
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
    return cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
def SobelY(img):
    return cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

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
    m = blockify(mag, cell_size, cell_size)
    a = blockify(angle, cell_size, cell_size)
    np.multiply(a[a > 180], -1, out=a[a > 180])
    np.add(a[a < -180], 360, out=a[a < -180])
    len = a.shape[0]
    histolist = np.zeros((len, 9))
    m = m.reshape((len, cell_size**2))
    a = a.reshape((len, cell_size**2))
    combi = np.zeros((len, cell_size**2, 2), dtype=float)
    for i in range(len):
        combi[i] = np.array((a[i], m[i])).T
        for a1, m1 in combi[i]:
            #split the magnitude between 2 angles, add each value to its slot in the histogram
            histogram = np.zeros(9, dtype=float)
            base = int((a1-(a1%20)) % 180 / 20)
            base2 = (base+1) % 9
            ratio = float((a1%20) / 20)
            ratio2 = 1 - ratio
            histogram[base] += m1 * ratio
            histogram[base2] += m1 * ratio2
        histolist[i] = histogram
    #add 9-bin histogram to list
    #histolist.append(np.histogram(cell, bins=9, range=(0,160))[0])
    #convert list into nested lists, then 3-dimensional numpy matrix (y by x by 9)
    h, w = angle.shape
    n1, n2 = int(h / cell_size), int(w / cell_size)
    histolist = histolist.reshape(n1, n2, 9)
    return histolist

def mag_normalize(mag):
    return

def show(img, mag):
    plt.subplot(1,2,1),plt.imshow(img,cmap = 'viridis')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,2,2),plt.imshow(mag,cmap = 'viridis')
    plt.title('Magnitude'), plt.xticks([]), plt.yticks([])
    plt.show()

img = cv2.imread('images/132x96.jpg',0)
img = crop(img, 8)
mag, angle = process(img)
'''
    img = cv2.imread('images/132x96.jpg',0)
    img = crop(img, 8)
    mag, angle = process(img)
    print(timeit.timeit("result = angle_hog(angle, mag, 8)",setup=setup, number=20)/20)
    result = angle_hog(angle, mag, 8)
    show(img, mag)
