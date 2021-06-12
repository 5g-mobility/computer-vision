import os
import argparse
from yolov5.dunas import Dunas
from yolov5.praiaBarra import PraiaBarra
from yolov5.riaAtiva import RiaAtiva
import torch
from yolov5.tasks import CeleryTasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam',
                         choices=['riaAtiva', 'ponteBarra', 'dunas'],
                         help='Camera to be used', required=True)
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='object confidence threshold')
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
    opt.nosave = True

    RABBITMQ_IP = os.environ.get("RABBITMQ_IP", None)

    if not RABBITMQ_IP:
        print("No RABBITMQ_IP ENV detected!")
        quit()

    # Info of url and others can be passed here to Celery
    celery_instance = CeleryTasks(RABBITMQ_IP)

    if opt.cam == 'riaAtiva':
        location = RiaAtiva(celery_instance)
        opt.weights = './yolov5/weights/best-riaAtiva.pt'
        opt.source = 'rtsp://pei:5g-mobix@10.0.19.201:554'
        
        
        
    elif opt.cam == 'ponteBarra':
        location = PraiaBarra(celery_instance)
        opt.weights = './yolov5/weights/best-ponte.pt'
        opt.source = 'rtsp://pei:5g-mobix@10.0.19.203:554'

    elif opt.cam == 'dunas':
        location = Dunas(celery_instance)
        opt.weights = './yolov5/weights/best-duna.pt'
        opt.source = 'rtsp://pei:5g-mobix@10.0.19.202:554'

    with torch.no_grad():
        location.detect(opt)

