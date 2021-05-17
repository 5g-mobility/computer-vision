import matplotlib.pylab as plt
import matplotlib as matlib
import cv2

import numpy as np

matlib.use('TkAgg')
image = cv2.imread('frame16.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(image)
plt.show()
