from camera import *
from tasks import CeleryTasks

class PraiaBarra(Camera):

    def __init__(self, celery):
        road_area = [([(1382.5, 546.2), (107, 872), (642, 1296), (1421.8, 546.2)],
                          [(107, 860), (0, 872), (0, 1296), (642, 1296)])]
        self.celery = celery
        model = "../sensor_fusion/ponte.pkl"
        super().__init__(road_area)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/best-ponte.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='rtsp://pei:5g-mobix@10.0.19.203:554', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.40, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()

    celery = CeleryTasks()

    barra =  PraiaBarra(celery)

    with torch.no_grad():
        barra.detect(opt)
