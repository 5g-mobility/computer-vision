import math
import cv2
import datetime
import argparse
#from easyocr import Reader
import re
from dunas import Dunas
from praiaBarra import PraiaBarra
from riaAtiva import RiaAtiva
import torch
import os


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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--rabbit_mq_url', help='URL of RabbitMQ', required=True, type=str)
    # parser.add_argument('--cam',
    #                     choices=['riaAtiva', 'ponteBarra', 'dunas'],
    #                     help='Camera to be used', required=True)
    # args = parser.parse_args()
    # opt = AttributeDict()

    # opt.config_deepsort = 'deep_sort_pytorch/configs/deep_sort.yaml'
    # opt.conf_thres = 0.4
    # opt.iou_thres = 0.5
    # opt.classes = [0]
    # opt.view_img = False
    # opt.img_size = 640
    # opt.device = 'cpu'
    # opt.augment = False
    # opt.agnostic_nms = False
    # opt.weights = './yolov5/weights/best-riaAtiva.pt'
    
    # pid = os.getpid()
    # #if args.cam == 'riaAtiva':
    # location = RiaAtiva()
    #     #opt.weights = './yolov5/weights/best-riaAtiva.pt'
    # # elif args.cam == 'ponteBarra':
    # #     location = PraiaBarra()
    # #     opt.weights = './yolov5/weights/best-ponte.pt'
    # # elif args.cam == 'dunas':
    # #     location = Dunas()
    # #     opt.weights = './yolov5/weights/best-duna.pt'
    # opt.pid = os.getpid()
    # with torch.no_grad():
    #     location.detect(opt)

    parser = argparse.ArgumentParser()
    parser.add_argument('--rabbit_mq_url', help='URL of RabbitMQ', required=True, type=str)
    parser.add_argument('--cam',
                         choices=['riaAtiva', 'ponteBarra', 'dunas'],
                         help='Camera to be used', required=True)
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    #parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    pid = os.getpid()

    if opt.cam == 'riaAtiva':
        location = RiaAtiva()
        opt.weights = './yolov5/weights/best-riaAtiva.pt'
    elif opt.cam == 'ponteBarra':
        location = PraiaBarra()
        opt.weights = './yolov5/weights/best-ponte.pt'
    elif opt.cam == 'dunas':
        location = Dunas()
        opt.weights = './yolov5/weights/best-duna.pt'

    opt.pid = os.getpid()



    with torch.no_grad():
        location.detect(opt, pid)

