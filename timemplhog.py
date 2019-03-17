import timeit

setup = '''
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage.feature import hog
from skimage import data, exposure


image = cv2.imread('images/132x96.jpg',0)
'''

print(timeit.timeit('hog_image = skimage.feature.hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),  feature_vector=True, block_norm="L2-Hys")',setup=setup, number=100)/100)