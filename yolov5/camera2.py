
import math
import torch
from utils.bbox import box_center, draw_boxes, compute_color_for_labels
import argparse
from pathlib import Path
from utils.general import increment_path
from utils.torch_utils import select_device, time_synchronized, load_classifier
from utils.general import set_logging, check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, scale_coord, xyxy2xywh_no_tensor
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random
import time
from utils.plots import plot_one_box, color_list
import cv2
import matplotlib.path as mpltPath
from tracker import Tracker, Detection
from dataObject import DataObject

IMG_SIZE = 2304, 1296

class Camera:

    def __init__(self, road_area=None):
        self.source = "../video/video_10s.mp4"
        self.road_area = road_area if road_area else [([(0, 0), (0, 0), (0, 0), (0, 0)])]
        self.mplt_path = [mpltPath.Path(area) for area in self.road_area]
        self.max_distance_between_points = 30
        self.ppm = 10
        self.fps = None
        self.tracker = Tracker(

        distance_function= self.euclidean_distance,
        distance_threshold= self.max_distance_between_points,
    )
    

    def estimateSpeed(self, location1, location2):
    
        """ calculate objetct speed """
        d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
        # ppm = location2[2] / carWidht
        # real = 100
        # 599.923537

        d_meters = d_pixels / self.ppm
        #print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))

        speed = d_meters * self.fps * 3.6
        return speed
	



    def process_tracking_data(self, tracked_objects, img, im0):
        
        bbox = []
        bbox_xyxy = []
        track_data = []
        
        for obj in tracked_objects:

            if obj.previous_detection:

                last_xyxy = [y  for b in obj.previous_detection.points.tolist() for y in b]            

            for box in obj.last_detection.points.tolist():
                for x in box:
                    bbox.append(x)

            bbox_xyxy.append(bbox)
            print(bbox)

            xywh = xyxy2xywh_no_tensor(bbox)

            track_data.append(
                DataObject(xywh, obj.last_detection.scores[0], 
                obj.last_detection.scores[1], 
                self.estimateSpeed(xywh, xyxy2xywh_no_tensor(last_xyxy) )))

            bbox = []
            
        



        #print(torch.tensor(bbox_xyxy))
        bbox_xyxy = scale_coord(
        img.shape[2:], torch.tensor(bbox_xyxy), im0.shape).round()
    
        # objects_detection = [obj.last_detection.points.tolist() for obj in tracked_objects]
        # bbox_xyxy = [obj for x in objects_detection.points for obj in x ]
        identities = [obj.id for obj in tracked_objects]

        bbox_xywh = xyxy2xywh(bbox_xyxy)

        return bbox_xyxy, identities, track_data

    def euclidean_distance(self, detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

    def detect(self, opt, save_img=False):
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)


        self.fps = dataset.fps

        print(self.fps)

        return


        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)


            norfair_detections = []

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f'{n} {names[int(c)]}s, '  # add to string
                        print("s: ", s)

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        
                        bbox = np.array([
                        [xyxy[0].item(), xyxy[1].item()],
                        [xyxy[2].item(), xyxy[3].item()] ])


                        norfair_detections.append(Detection(bbox, np.array([conf, cls])))


                tracked_objects = self.tracker.update(detections=norfair_detections)

                if tracked_objects:

                    bbox_xyxy,identities, track_data =  self.process_tracking_data(tracked_objects, img, im0)

                    print(len(bbox_xyxy))
                    #draw_boxes(im0, bbox_xyxy, indetities)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
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


    

    #print(camera.source)
    camera =  Camera()
    # dunas = Camera()

    with torch.no_grad():
        camera.detect(opt)
