from camera import *
from tasks import CeleryTasks

class RiaAtiva(Camera):

    def __init__(self, celery):

        road_area = [ [(895, 194), (1002, 1296), (2290, 1296), (990, 194)],
                          [(1762, 810), (2165, 1296), (2304, 1296), (2304, 984)] ]
        detect_area = [([890.2, 255.9], [1095.5, 267.0]),([885, 313.6],[1166.9, 335.2])]
        detect_dist = 14
        radar_id = 7
        self.celery = celery
        max_distance_between_points = 57
        model = "./sensor_fusion/ria.pkl"
        super().__init__(road_area, model, detect_area, detect_dist, radar_id, max_distance_between_points)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/best-riaAtiva.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='rtsp://pei:5g-mobix@10.0.19.201:554', help='source')  # file/folder, 0 for webcam
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

    ria =  RiaAtiva(celery)

    with torch.no_grad():
        ria.detect(opt)
