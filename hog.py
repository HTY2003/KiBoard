import cv2
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(threshold=np.inf)

# retrieve data from image
img = cv2.imread('download.jpg',0)

def crop(img, cell_size):
    #crop height and width to be divisble by 8
    excessh = len(img) - len(img) % cell_size
    excessw = len(img[0]) - len(img[0]) % cell_size
    img = img[:excessh, :excessw]
    return img

def process(img):
    #run img through Sobel operators (edge detection)
    gx, gy = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5), cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    # calculate magnitude and direction (in angles) of the image
    # magnitude is calculated using  (dx**2 + dy**2) // 2
    # gradient is calculated using arctan(dy / dx)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mag, angle = np.absolute(mag), np.absolute(angle)
    return mag, angle

def angle_hog(angle, cell_size):
    #filter angles into a 0-180deg 9-bin histogram
    histolist = []
    cell_list = [(angle[y:y+cell_size, x:x+cell_size]) for y in range(0, len(angle)-cell_size+1, cell_size) for x in range(0, len(angle[0])-cell_size+1, cell_size)]
    #make the gradients unsigned (negative angles < 180 = positive)
    for cell in cell_list:
        cell[cell > 180] *= -1
        cell[cell < -180] += 360
        cell = np.uint8(cell)
        #add 9-bin histogram to list
        histolist.append(np.histogram(cell, bins=9, range=(0,180))[0])
    #convert list into nested lists, then 3-dimensional numpy matrix (y by x by 9)
    n1, n2 = int(len(angle) / cell_size), int(len(angle[0]) / cell_size)
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

img = crop(img, 8)
mag, angle = process(img)
print(angle_hog(angle, 8))
show(img, mag)
