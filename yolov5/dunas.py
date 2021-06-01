from tasks import CeleryTasks
from camera import *


class Dunas(Camera):

    def __init__(self, celery):
        self.source = "./video/video_10s.mp4"
        #self.source = "rtsp://pei:5g-mobix@10.0.19.202:554"
        """
            top_left, top_right, bottom_left, bottom_right
        """
        road_area = [([(307.7, 114.2), (349.6, 106.5), (501.1, 477.5), (781.1, 479.2)])]
        self.celery = celery
        super().__init__(road_area = road_area)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/best-duna.pt', help='model.pt path')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)

    celery = CeleryTasks()

    dunas = Dunas(celery)

    with torch.no_grad():
        dunas.detect(args)
