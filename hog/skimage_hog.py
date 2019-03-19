import matplotlib.pyplot as plt
import cv2
from skimage import exposure
from skimage.feature import hog
from skimage.transform import resize

class HOG_Feature:
    def __init__(self, res, bins, cell, block):
        self.bins = bins
        self.res = res
        self.cell = cell
        self.block = block
    
    def vector(img, rsize=True)
        image = cv2.imread('images/' + img,0)
        if rsize == True: image = resize(image, (96, 128), anti_aliasing=True)
        return hog(image, orientations=self.bins, pixels_per_cell=(self.cell)*2,
                    cells_per_block=(self.block)*2, block_norm="L2", feature_vector=True)

    def visualize(img, rsize=True):
        image = cv2.imread('images/' + img,0)
        if rsize == True: image = resize(image, (96, 128), anti_aliasing=True)
        hog_image = hog(image, orientations=self.bins, pixels_per_cell=(self.cell)*2,
                    cells_per_block=(self.block)*2, block_norm="L2", 
                    visualize=True, feature_vector=True)[1]
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 12))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax2.axis('off')
        ax2.imshow(hog_image, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()