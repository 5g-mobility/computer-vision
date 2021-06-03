import cv2


def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("frame4.jpg", image)


if __name__ == '__main__':
    #getFirstFrame("./yolov5/runs/detect/exp/pei_5g-mobix_10.0.19.202_554.mp4")
    getFirstFrame("66.mp4")
