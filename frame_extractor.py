import cv2


def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success,image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


if __name__ == '__main__':
    #getFirstFrame("./yolov5/runs/detect/exp/pei_5g-mobix_10.0.19.202_554.mp4")
    getFirstFrame("./yolov5/runs/detect/exp38/pei_5g-mobix_10.0.19.201_554.mp4")
