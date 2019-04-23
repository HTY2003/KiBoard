import matplotlib.pyplot as plt
import cv2
from skimage import exposure
from skimage.feature import hog
from skimage.transform import resize

class HOG_Feature:
    def __init__(self, res=(96,128), bins=8, cell=8, block=2):
        self.bins = bins
        self.res = res
        self.cell = cell
        self.block = block

    def hog(self, img, rsize=True):
        image = cv2.imread('images/' + img,0)
        if rsize: image = resize(image, self.res, anti_aliasing=False)

        feature_vector = hog(image, orientations=self.bins, 
                            pixels_per_cell=(self.cell, self.cell),
                            cells_per_block=(self.block,self.block),
                            block_norm="L2", feature_vector=True)

        return feature_vector

    def show(self, img, rsize=True):
        image = cv2.imread('images/' + img,0)
        if rsize: image = resize(image, self.res, anti_aliasing=False)

        hog_image = hog(image, orientations=self.bins, 
                            pixels_per_cell=(self.cell, self.cell),
                            cells_per_block=(self.block,self.block),
                            block_norm="L2", feature_vector=True,
                            visualize=True)[1]
                            
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 12))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax2.axis('off')
        ax2.imshow(hog_image, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()
