import cv2
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)

def blockify(arr, y, x, z):
    '''
    Chop matrix into blocks of nrows by ncols
    a.k.a turn an matrix of shape y,x,z into y/nrows,x/ncols,8,8,z
    '''
    h, w = arr.shape[:2]
    return (arr.reshape(h//y, y, -1, x, z)
           .swapaxes(1,2)
           .reshape(-1, y, x, z))

def crop(img, cell_size):
    '''
    Crop the matrix to be divisble by cell size in height and width
    Note: not needed in final product
    '''
    h, w = img.shape
    excessh = h - h % cell_size
    excessw = w - w % cell_size
    img = img[:excessh, :excessw]
    return img

def process(img):
    '''
    1) Run image through Sobel operators (edge detection)
    2a) Calculate magnitude and direction (in angles) of the image
    2b) mag: (dx**2 + dy**2) // 2, angle: arctan(dy / dx)
    3) Return 2 matrices of magnitude and angle
    '''
    mag, angle = cv2.cartToPolar(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3), cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3), angleInDegrees=True)
    mag, angle = mag.round(2), np.int16(angle)
    return mag, angle

def angle_hog(angle, mag, cell_size):
    '''
    Calculate histogram from angle and magnitude, with each histogram consisting of square cell of cell_size
    Note: Look in code for more details
    '''
    angle[angle > 180] -= np.int16(180)
    # 96,128 + 96,128 => 192,8,8,2
    combi = blockify(np.vstack(([mag.T], [angle.T])).T, cell_size, cell_size, 2)
    hxw = combi.shape[0]
    # 192,8,8,2 => 192,64,2
    combi = combi.reshape(hxw, cell_size**2, 2)
    histolist = np.zeros((hxw, 9))
    #create a list of histograms with 9 bins
    for i in range(hxw):
        for m,a in combi[i]:
            histogram = np.zeros(9)
            base = int((a-(a%20)) % 180 / 20)
            base2 = (base+1) % 9
            m1 = m * float((a%20) / 20)
            m2 = m - m1
            histogram[base] += m2
            histogram[base2] += m1
        histolist[i] = histogram
    n1 = int(angle.shape[0] / cell_size)
    n2 = int(hxw / n1)
    #192,9 => 12,16,9
    histolist = histolist.reshape(n1, n2, 9)
    return histolist

def mag_normalize(histo, block_size):
    '''
    Normalize histograms with blocks of 2x2 cells
    '''
    h,w,z = histo.shape
    newhistolist = np.zeros((h-1, w-1, (block_size**2)*9))
    for y in range(h-2):
        for x in range(w-2):
            block = histo[y:y+2,x:x+2].flatten()
            newhistolist[y,x] = np.linalg.norm(block) * block
    return newhistolist.flatten()

img = cv2.imread('images/132x96.jpg',0)
img = crop(img, 8)
mag, angle = process(img)
result = mag_normalize(angle_hog(angle, mag, 8), 2)
