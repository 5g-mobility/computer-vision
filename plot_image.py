import matplotlib.pylab as plt
import cv2
import matplotlib.path as mpltPath
import argparse
import numpy as np



def main(image):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.imshow(image)
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='source')
    args = parser.parse_args()

    main(args.image)