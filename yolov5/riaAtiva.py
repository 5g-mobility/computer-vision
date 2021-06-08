from camera import *
from tasks import CeleryTasks
import numpy as np

class RiaAtiva(Camera):

    def __init__(self, celery):

        road_area = [ [(895, 194), (1002, 1296), (2288, 1296), (988, 194)],
                          [(1762, 810), (2165, 1296), (2304, 1296), (2304, 984)] ]
        detect_area = [([890.2, 255.9], [1095.5, 267.9]),([885, 313.6],[1166.9, 337.0])]
        detect_dist = 14
        radar_id = 7
        self.celery = celery
        model = "./sensor_fusion/ria.pkl"
        super().__init__(road_area, model, detect_area, detect_dist, radar_id)


        # def rescale_coords(self,point, img):
        # x, y = point
        # img_y, img_x = img.shape[:2]
        # n_img_x,n_img_y = IMG_SIZE

        # return x * (img_x/n_img_x), y * (img_y/n_img_y)


    # def calibrate_geoCoods(self, coords, geo_coords):

    #     lat, lon =  geo_coords


    #     coords_rescale = self.rescale_coords(coords, )


    #     p = np.array([lat, lon])

    #     line1 = np.array()

    #     line2 = np.array()


    #     return lat, lon

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
