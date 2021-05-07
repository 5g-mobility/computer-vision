import matplotlib.pylab as plt
import cv2

import numpy as np

image = cv2.imread('frame1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)
plt.show()