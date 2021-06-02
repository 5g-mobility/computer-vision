import queue
import re
import sys
import threading
from datetime import datetime
from easyocr import Reader
import random

# sys.path.insert(0, './yolov5')
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_coords, apply_classifier, scale_coords, \
    increment_path, set_logging, xyxy2xywh, check_imshow, scale_coord
from utils.torch_utils import select_device, time_synchronized, load_classifier
from models.experimental import attempt_load
from tracker import Detection, Tracker
from utils.plots import plot_one_box
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import json
import matplotlib.path as mpltPath
from numpy import random
import numpy as np
import math
import pickle
from dataObject import DataObject

class Camera:

    def __init__(self, road_area=None):
        self.source = "./video/video_10s.mp4"
        self.road_area = road_area if road_area else [([(0, 0), (0, 0), (0, 0), (0, 0)])]
        
        for area in self.road_area:
            print(area)
        self.mplt_path = [mpltPath.Path(area) for area in self.road_area]
        self.time_objects = {}  # Object JSON Data to be sent to celery regarding one timestamp
        self.q = queue.Queue()  # Queue of frames to be OCR
        self.old_ids = set()  # Old Deep Sort Object Ids -> maybe this can be cleaned up every x time
        self.fps = None
        self.max_distance_between_points = 30
        self.ppm = 10
        #self.mapping = self.initialize_mapping_model()
        self.tracker = Tracker(

        distance_function= self.euclidean_distance,
        distance_threshold= self.max_distance_between_points,
    )
        #threading.Thread(target=self.process_data, daemon=True).start()

    @property
    def palette(self):
        """ bounding boxes colors """
        return (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


    def initialize_mapping_model(self):

        with open("./sensor_fusion/ridge.pkl", 'rb') as file:
            pickle_model = pickle.load(file)

        return pickle_model
    

    
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
            bbox = []  
            track_data.append(
                DataObject(xyxy2xywh(bbox), obj.last_detection.scores[0], 
                obj.last_detection.scores[1], 
                self.estimateSpeed(xyxy2xywh(bbox),xyxy2xywh(last_xyxy) )))


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


    # check if object is inside road
    def inside_road(self, x, y):
        """ check if object is inside road or not """

        return any([path.contains_point((x, y), radius=1e-9) for path in self.mplt_path])


    # def euclidean_distance(detection, tracked_object):
        
    #     """ caculate euclidean distance for tracking  """

    #     return np.linalg.norm(detection.points - tracked_object.estimate)


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
	

    def box_center(self, *xyxy):
        """ calculate bbox center based xy points """

        x1, y1, x2, y2 = xyxy

        return (int(x1) + int(x2)) / 2, (int(y1) + int(y2)) / 2

    def isMotocycle(self, center_x, center_y):
        """ check if detected object is an motocycle

            if ouside of road - cyclist
            else - motocycle
        """

        if (self.inside_road(center_x, center_y)):
            return True

        return False

    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)

    def draw_boxes(self, img, track_objects, identities, offset=(0, 0)):
        
        """ draw bbox of each object in frame """ 

        for i, box in enumerate(track_objects):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = self.compute_color_for_labels(id)
            label = '{}{:d}'.format("", id)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 +
                                    t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img
    


    
    def detect(self, opt):
        weights, view_img, save_txt, imgsz = opt.weights, opt.view_img, opt.save_txt, opt.img_size
        save_img = not opt.nosave and not self.source.endswith('.txt')  # save inference images
        webcam = self.source.isnumeric() or self.source.endswith('.txt') or self.source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        print(weights)
        img_size = (2304, 1296)  # x, y

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

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
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

        self.fps =  dataset.fps

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        names.append('motocycle')

        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()

        old_im0s = np.array([], [])

        for path, img, im0s, vid_cap in dataset:
            # if np.array_equal(im0s, old_im0s):
            #     time.sleep(0.01)
            #     continue

            # old_im0s = im0s

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)

            
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
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                if len(det):
                    # Rescale boxes from img_size to im0 size
  

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if webcam:  # batch_size >= 1
                            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                        else:
                            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)


                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)


                        bbox = np.array([
                        [xyxy[0].item(), xyxy[1].item()],
                        [xyxy[2].item(), xyxy[3].item()] ])

                        norfair_detections.append(Detection(bbox, np.array([conf, cls])))


            tracked_objects = self.tracker.update(detections=norfair_detections)

            #print(tracked_objects)

            if tracked_objects:
                bbox_xyxy,identities, track_data =  self.process_tracking_data(tracked_objects, img, im0)

                for obj in track_data:

                    self.send_data(obj, c=cls, names=names, frame=im0)


                self.draw_boxes(im0, bbox_xyxy, identities)


            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


                        # if save_txt:  # Write to file
                        #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        #     line = (cls, *xyxy, conf) if opt.save_conf else (
                        #         cls,)  # label format
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * len(line)).rstrip() % line + '\n')

 

                        # if save_img or view_img:  # Add bbox to image
                        #     label = f'{names[int(cls)]} {conf:.2f}'
                        #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')

        # except TimeoutError as time_err:
        #     print(f'Timeout error: {time_err}')
        #     flag_repeat_LoadStream = True
        #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        # except Exception as e:
        #     print(f'Exception occur when detecting: {e}')

    
        # # Initialize
        # device = select_device(opt.device)
        # half = device.type != 'cpu'  # half precision only supported on CUDA

        # # Load model yolo5
        # model = torch.load(weights, map_location=device)[
        #     'model'].float()  # load to FP32
        # model.to(device).eval()
        # if half:
        #     model.half()  # to FP16

        # view_img = True
        # cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadStreams(self.source, img_size=imgsz)

        # # Get names and colors
        # names = model.module.names if hasattr(model, 'module') else model.names
        # names.append('motocycle')
        # # Run inference
        # t0 = time.time()
        # img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        # # run once
        # _ = model(img.half() if half else img) if device.type != 'cpu' else None

        # for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        #     img = torch.from_numpy(img).to(device)
        #     img = img.half() if half else img.float()  # uint8 to fp16/32
        #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
        #     if img.ndimension() == 3:
        #         img = img.unsqueeze(0)

        #     # Inference
        #     t1 = time_synchronized()
        #     pred = model(img, augment=opt.augment)[0]

        #     # Apply NMS
        #     pred = non_max_suppression(
        #         pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        #     t2 = time_synchronized()

        #     print("pred : ", pred)
        #     # Process detections
        #     for i, det in enumerate(pred):  # detections per image

        #         # det - pytorch tensor (matriz)

        #         print("det: ", det)
        #         print(det[:, -1])

        #         # det- detections

        #         p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()

        #         s += '%gx%g ' % img.shape[2:]  # print string

        #         if len(det):
        #             # Rescale boxes from img_size to im0 size
        #             det[:, :4] = scale_coords(
        #                 img.shape[2:], det[:, :4], im0.shape).round()

        #             # Print results
        #             for c in det[:, -1].unique():
        #                 print("c: ", c)
        #                 print("name: ", names[int(c)])
        #                 # c - classe
        #                 # n - numero de objetos detetados para a classe
        #                 # names - dicionario com o nome das classes
        #                 n = (det[:, -1] == c).sum()  # detections per class
        #                 s += '%g %ss, ' % (n, names[int(c)])  # add to string

        #             bbox_xywh = []
        #             confs = []
        #             classes = []
        #             print(det[:, -1])

        #             # Adapt detections to deep sort input format
        #             for *xyxy, conf, cls in det:
        #                 # cls - classe
        #                 x_c, y_c, bbox_w, bbox_h = self.bbox_rel(*xyxy)
        #                 obj = [x_c, y_c, bbox_w, bbox_h]

        #                 bbox_xywh.append(obj)
        #                 confs.append([conf.item()])

        #                 print("cls : ", cls)

        #                 classes.append(cls)

        #             xywhs = torch.Tensor(bbox_xywh)
        #             confss = torch.Tensor(confs)

        #             print("antes: ", bbox_xywh)
        #             print("depois: ", xywhs)

        #             # im0 - imagem

        #             # Pass detections to deepsort
        #             #outputs = deepsort.update(xywhs, confss, im0, classes)
        #             #print("outputs: ", outputs)
        #             # draw boxes for visualization
        #             if len(outputs) > 0:
        #                 bbox_xyxy = outputs[:, :4]
        #                 identities = outputs[:, -2]
        #                 classes = outputs[:, -1]

        #                 self.draw_boxes(im0, bbox_xyxy, [names[int(c)] for c in classes], identities)

        #             # Write MOT compliant results to file
        #             if len(outputs) != 0:
        #                 for j, output in enumerate(outputs):
        #                     json_data = self.send_data(output, names)

        #                     print(json_data)

        #         else:
        #             deepsort.increment_ages()

        #         # Print time (inference + NMS)
        #         print('%sDone. (%.3fs)' % (s, t2 - t1))

        #         # Stream results
        #         if view_img:
        #             cv2.imshow(p, im0)
        #             if cv2.waitKey(1) == ord('q'):  # q to quit
        #                 raise StopIteration

        # print('Done. (%.3fs)' % (time.time() - t0))

    def bbox_rel(self, *xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

    def compute_color_for_labels(self, label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in self.palette]
        return tuple(color)

    def draw_boxes(self, img, bbox, classes, identities=None, offset=(0, 0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = self.compute_color_for_labels(id)
            cls = classes[i]

            label = '{}{:d}  {}'.format("", id, cls)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 +
                                     t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img

    def send_data(self, obj, c, names, frame):
        """
        Only send frames to this function if an object is detected to further processing.
        It is expected that deep_sort sends objects data with ids associated to them
        Only send cars, trucks etc if they are on the road and on radar zone
        """
        id = random.randint(1, 1000)  # Later on, this will be the deep sort id

        if id in self.old_ids:
            """
            If this object already appeared in older frames, it is no longer necessary to process. 
            If it is a car, truck, bicycle, etc in the radar zone, it is only necessary to sensor fusion the first 
            appearance in the radar zone. If it is a person in bike lane or something, we only report the first time 
            that person it is seen.
            """
            return
        else:
            self.old_ids.add(id)

        bbox_left = obj.xyxy[0]
        bbox_top = obj.xyxy[1]
        bbox_w = obj.xyxy[2]
        bbox_h = obj.xyxy[3]
        # id

        center_x, center_y = self.box_center(obj.xyxy)
        # bike
        if obj.cls == 1 and self.isMotocycle(center_x, center_y):
            obj.cls = len(names) - 1

        #lat, lon = self.mapping.predict()
        lat, lon  =1, 1
        # id should be the id from deep sort
        # box_w and other stuff is not needed, instead of the class maybe send the EVENT_TYPE AND EVENT_CLASS ->
        # Dps fala comigo Miguel, ass Hugo
        data = json.dumps({"class": names[int(obj.cls)],"lat": lat,
        "long": lon, "speed": obj.velocity
                           , "id": id})

        self.q.put((data, frame[50:125, 1725:2250]))

        return data

    def process_data(self):
        """Processes time of the frame and, accordingly with the data, it will send """
        while True:
            json, image_time = self.q.get()
            now = datetime.now()
            time_from_image = Reader(['en']).readtext(image_time, detail=0)
            res = re.findall("\d{2}", time_from_image[0])
            try:
                date = datetime.strptime("{}{} {}".format(res[0], res[1], " ".join(res[2:])),
                                         "%Y %m %d %H %M %S")
                if date.year != now.year:
                    print("Bad Year processed, was: {}".format(date.year))
                    date.year = now.year
                if date.month != now.month:
                    print("Bad Month processed, was: {}".format(date.month))
                    date.month = now.month
                if date.day != now.day:
                    # Maybe it's better to check if the difference between the days is higher than two
                    print("Bad Day processed, was: {}".format(date.day))
                    date.day = now.day
                if date.hour != now.hour:
                    # Maybe it's better to check if the difference between the hours is higher than two
                    print("Bad Hour processed, was: {}".format(date.hour))
                    date.hour = now.hour

                print(date)

                json['date'] = date

                if date not in self.time_objects:
                    # Sending old datetime objects data
                    for key in self.time_objects:
                        for json in self.time_objects[key]:
                            self.celery.send_data(json)
                    self.time_objects[date] = [json]

            except ValueError:
                print("Error while parsing date")

            self.q.task_done()




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

    dunas = Camera()

    with torch.no_grad():
        dunas.detect(args)