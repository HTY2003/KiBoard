import os

from hog.skimage_hog import HOG_Feature
from hog.lda import WeakLearner
import numpy as np

hog = HOG_Feature()
model = WeakLearner()
def find_and_train(list_of_images, another_list_of_images):
    list_of_vectors =  np.array([hog.hog(img, rsize=True) for img in list_of_images])

    another_list_of_vectors =  np.array([hog.hog(img, rsize=True) for img in another_list_of_images])

    model.train(list_of_vectors, another_list_of_vectors)

imglist1 = os.listdir("images/")
imglist2 = os.listdir("images/")
find_and_train(imglist1, imglist2)