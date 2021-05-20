import math
import cv2
import datetime
import argparse
from easyocr import Reader
import re
from dunas import Dunas
from praiaBarra import PraiaBarra
from riaAtiva import RiaAtiva
import torch


class AttributeDict(dict):
    __slots__ = ()
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def connectToCam(url, user, pw, port):
    url = "rtsp://{}:{}@{}:{}".format(user, pw, url, port)
    cap = cv2.VideoCapture(url, cv2.CAP_ANY)
    return cap


def readFrames(url, user, pw, port=554):
    cap = connectToCam(url, user, pw, port)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while True:
        ret, frame = cap.read()
        if not ret:
            cap = connectToCam(url, user, pw, port)
            fps = cap.get(cv2.CAP_PROP_FPS)
            continue
        else:
            frameId = cap.get(1)
            if frameId % math.floor(fps) == 0:
                image_time = frame[50:125, 1725:2250]
                cv2.imshow("Current frame", image_time)
                time_from_image = Reader(['en']).readtext(image_time, detail=0)
                res = re.findall("\d{2}", time_from_image[0])
                date = datetime.datetime.strptime("{}{} {}".format(res[0], res[1], " ".join(res[2:])),
                                                  "%Y %m %d %H %M %S")
                print(date)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rabbit_mq_url', help='URL of RabbitMQ', required=True, type=str)
    parser.add_argument('--cam',
                        choices=['riaAtiva', 'ponteBarra', 'dunas'],
                        help='Camera to be used', required=True)
    args = parser.parse_args()
    opt = AttributeDict()

    opt.config_deepsort = 'deep_sort_pytorch/configs/deep_sort.yaml'
    opt.conf_thres = 0.7
    opt.iou_thres = 0.5
    opt.classes = [0]
    opt.view_img = False
    opt.img_size = 640
    opt.device = 'cpu'
    opt.augment = False
    opt.agnostic_nms = False
    if args.cam == 'riaAtiva':
        location = RiaAtiva()
        opt.weights = 'yolov5/weights/best-riaAtiva.pt'
    elif args.cam == 'ponteBarra':
        location = PraiaBarra()
        opt.weights = 'yolov5/weights/best-ponte.pt'
    elif args.cam == 'dunas':
        location = Dunas()
        opt.weights = 'yolov5/weights/best-duna.pt'

    with torch.no_grad():
        location.detect(opt)
