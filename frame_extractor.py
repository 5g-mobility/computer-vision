import cv2


def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        cv2.imwrite("frame1.jpg", image)


if __name__ == '__main__':
    getFirstFrame("./yolov5/runs/detect/exp25/pei_5g-mobix_172.17.10.22_554.mp4")
