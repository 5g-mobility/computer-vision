
import math
from os import times
import torch
from utils.bbox import box_center, draw_boxes, draw_detection_area, is_inside_area
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
import threading
from utils.plots import plot_one_box, color_list
import cv2
import matplotlib.path as mpltPath
from tracker import Tracker, Detection 
from dataObject import DataObject
import json
import queue
import pickle
import pandas as pd
import re
from datetime import datetime
from easyocr import Reader
from datetime import timezone, timedelta
import matplotlib.pylab as plt



IMG_SIZE = 2304, 1296

class Camera:

    def __init__(self, road_area=None, model_path=None, detect_area = None, detect_dist=0, radarId = None):
        self.road_area = road_area if road_area else [([(0, 0), (0, 0), (0, 0), (0, 0)])]
        self.is_road_scale = False
        self.max_distance_between_points = 65
        self.ppm = 10
        self.fps = None
        self.radarId = radarId
        self.detect_area =  detect_area
        self.detect_dist = detect_dist
        self.mapping = self.initialize_mapping_model(model_path)
        self.time_objects = {}  # Object JSON Data to be sent to celery regarding one timestamp
        self.q = queue.Queue()  # Queue of frames to be OCR
        self.old_ids = set()  # Old Deep Sort Object Ids -> maybe this can be cleaned up every x time
        self.tracker = Tracker(

        distance_function= self.euclidean_distance,
        distance_threshold= self.max_distance_between_points,
        initialization_delay = 2
    )
        self.reader = Reader(['en'])
        threading.Thread(target=self.process_data, daemon=True).start()

    def estimateSpeed(self, time):
    
        """ calculate objetct speed """

        return self.detect_dist/(time*0.001) * 3.6

    def calibrate_geoCoods(self, coords):

        lat, lon =  coords

        return lat, lon
	
    def send_data(self, obj, names, gn):
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


        xywh = xyxy2xywh(obj.xyxy.view(1, 4)).view(-1)
        


        center_x, center_y = xywh.tolist()[:2]

        is_inside = self.inside_road(center_x, center_y)

        # bike
        if obj.cls == 1 and self.isMotocycle(is_inside):

            obj.cls = len(names) - 1
        
        
        xywh_norm = (xyxy2xywh(obj.xyxy.view(1, 4)) / gn).view(-1).tolist()

        #lat, lon =self.calibrate_geoCoods( (center_x, center_y ), self.mapping.predict(np.asarray([xywh_norm[0:2]])).tolist()[0])
        lat, lon = self.mapping.predict(np.asarray([xywh_norm[0:2]])).tolist()[0]

        if obj.is_stopped:

            data = {"id": obj.idx, "class": names[int(obj.cls)],"lat": lat, "long": lon, "speed": 0 , "inside_road": is_inside, "is_stopped": True, "radarId": self.radarId}

        elif obj.cls in [0,4,5,6,7,9]:
             data = {"id": obj.idx, "class": names[int(obj.cls)],"lat": lat, "long": lon, "inside_road": is_inside, "is_stopped": True, "radarId": self.radarId}

        else:
            data = {"id": obj.idx, "class": names[int(obj.cls)],"lat": lat, "long": lon, "speed": obj.velocity, "inside_road": is_inside,  "is_stopped": False, "radarId": self.radarId}



        # id should be the id from deep sort
        # box_w and other stuff is not needed, instead of the class maybe send the EVENT_TYPE AND EVENT_CLASS ->
        # Dps fala comigo Miguel, ass Hugo
            

        self.q.put((data, obj.frame[50:125, 1725:2250]))


    def process_tracking_data(self, tracked_objects, img, im0, times):
        
        bbox_xyxy = []
        track_data = []
        idx = []
        scores = []
        last_xyxy = []
        n_stops = []


        for obj in tracked_objects:

            

            if obj.previous_detection:

                last_xyxy.append([y  for b in obj.previous_detection.points.tolist() for y in b] )           

            bbox_xyxy.append([x for box in obj.last_detection.points.tolist() for x in box])

            scores.append(obj.last_detection.scores)

            idx.append(obj.id)

            n_stops.append( obj.n_stop)


        bbox_xyxy = scale_coords(
        img.shape[2:], torch.tensor(bbox_xyxy), im0.shape).round()

        for i, box in enumerate(bbox_xyxy):
            

            is_stopped = n_stops[i] > 3
            
            if int(scores[i][1]) == 0 and not tracked_objects[i].arealy_tracked: #Person
                
                track_data.append(
                    
                DataObject(idx[i], box, scores[i][1], scores[i][1], is_stopped ,velocity=0,  frame = im0, person=True))

                tracked_objects[i].arealy_tracked = True

                
            
            else:

                center_x, center_y = box_center(box)

                if tracked_objects[i].cross_line != True and is_inside_area((center_x, center_y), self.detect_area[0],self.detect_area[1]) :
                    

        
                    tracked_objects[i].cross_line = True
                    tracked_objects[i].init_time = times
                    tracked_objects[i].frame = im0
                    

                elif tracked_objects[i].cross_line == True and not is_inside_area((center_x, center_y), self.detect_area[0],self.detect_area[1]):
                    
                    
                
        
                    direction = -1 if tracked_objects[i].last_detection.points[0][1] - tracked_objects[i].previous_detection.points[0][1] > 0 else 1

                    speed = round(self.estimateSpeed(times - tracked_objects[i].init_time) * direction, 2)

                    
                    track_data.append(
                        
                    DataObject(idx[i], box, scores[i][1], scores[i][1], is_stopped, 
                        speed , tracked_objects[i].frame ))


                    tracked_objects[i].cross_line = False
                    tracked_objects[i].init_time = None
                    tracked_objects[i].frame = None

                    continue

                track_data.append(
                    
                DataObject(idx[i], box, scores[i][1], scores[i][1], is_stopped, frame = im0))

    
        return bbox_xyxy, track_data
    

    def initialize_mapping_model(self, model_path = None):

        if model_path:
            with open(model_path, 'rb') as file:
                pickle_model = pickle.load(file)

        else:
            with open("../sensor_fusion/ridge.pkl", 'rb') as file:
                pickle_model = pickle.load(file)

        return pickle_model

    def inside_road(self, x, y):
        """ check if object is inside road or not """

        return any([path.contains_point((x, y), radius=1e-9) for path in self.mplt_path])



    def isMotocycle(self, is_inside):
        """ check if detected object is an motocycle
    
            if ouside of road - cyclist
            else - motocycle
        """

        if (is_inside):
            return True

        return False

    def rescale_coords(self,point, img):
        x, y = point
        img_y, img_x = img.shape[:2]
        n_img_x,n_img_y = IMG_SIZE

        return x * (img_x/n_img_x), y * (img_y/n_img_y)






    def process_data(self):
        """Processes time of the frame and, accordingly with the data, it will send """

        while True:

            print("aki")

            try:
                json, image_time = self.q.get(block= True, timeout=5)


            except:
                

                print("Timeout ...")

                keys_to_del = []
                if self.time_objects:  

                    for key in self.time_objects:
                            
                        #self.celery.send_data(self.time_objects[key])
                        keys_to_del.append(key)
                            
                    for k in keys_to_del:
                        del self.time_objects[k]

                continue



            now = datetime.utcnow()
            time_from_image = self.reader.readtext(image_time, detail=0)
            res = re.findall("\d{2}", time_from_image[0])
            try:
                date = datetime.strptime("{}{} {}".format(res[0], res[1], " ".join(res[2:])),
                                            "%Y %m %d %H %M %S") - (timedelta(hours=1)) 
                
                
                if '203' in self.source and json['speed'] > 0:
                    date -= timedelta(seconds=1)

                if '201' in self.source and json['class'] in ['car', 'truck', 'motocycle']:
                    json['speed']= json['speed'] * -1
                    if json['speed'] > 0:
                        date -= timedelta(seconds=2)

                if date.year != now.year:
                    print("Bad Year processed, was: {}".format(date.year))
                    date = datetime(now.year,date.month,  date.day, date.hour, date.minute, date.second )

                if date.month != now.month:
                    print("Bad Month processed, was: {}".format(date.month))

                    date = datetime(date.year,now.month,  date.day, date.hour, date.minute, date.second )
                
                if date.day != now.day:
                  
                    print("Bad Day processed, was: {}".format(date.day))
                    date = datetime(date.year,date.month,  now.day, date.hour, date.minute, date.second )
               



                then = datetime(date.year,date.month,  date.day, date.hour,0, 0)
                actual = datetime(date.year,date.month,  date.day, now.hour,0, 0)
                if abs((then - actual).total_seconds()) >= 2*3600:
                    # Maybe it's better to check if the difference between the hours is higher than two
                    print("Bad Hour processed, was: {}".format(date.hour))
                    date = datetime(date.year,date.month,  date.day, now.hour, date.minute, date.second )
                    
                then = datetime(date.year,date.month,  date.day, date.hour,date.minute, 0)
                actual = datetime(date.year,date.month,  date.day, now.hour,now.minute, 0)

                if abs((then - actual).total_seconds()) >= 2*60:
                    print("Bad Minute processed, was: {}".format(date.hour))
                    date = datetime(date.year,date.month,  date.day, date.hour, now.minute, date.second )

                then = datetime(date.year,date.month,  date.day, date.hour,date.minute, date.second)
                actual = datetime(date.year,date.month,  date.day, now.hour,now.minute, now.second)

                if abs((then - actual).total_seconds()) >= 20:
                    print("Bad Second processed, was: {}".format(date.hour))
                    self.q.task_done()
                    continue

                

                json['date'] = str(date)

                print(date)
                print(json)

                if date not in self.time_objects:

                    del_keys = []
                    # Sending old datetime objects data
                    for key in self.time_objects:
                       
            
                        #self.celery.send_data(self.time_objects[key])
                        del_keys.append(key)
                         
                    for k in del_keys:
                        del self.time_objects[k]

                        
                    self.time_objects[date] = [json]
                else:
                    self.time_objects[date].append(json)

        

            except ValueError:
                print("Error while parsing date")

            self.q.task_done()



    


    def euclidean_distance(self, detection, tracked_object):
        return np.linalg.norm(detection.points[0:2] - tracked_object.estimate[0:2])

    def rescale_detect_area(self, line, im0):

        p1, p2 = line

        return [round(x) for x in list(self.rescale_coords(p1, im0))], [round(x) for x in list(self.rescale_coords(p2, im0))]


    def detect(self, opt, save_img=False):
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        self.source = source
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        stream = False

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
            
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
            stream = True

        else:
            
            dataset = LoadImages(source, img_size=imgsz)


        self.fps = dataset.fps

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        names.append('motocycle')
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        old_im0s = np.array([], [])

        for path, img, im0s, vid_cap, times in dataset:
            #im0s -  imagem original
            # img - imagem resize


            if not self.is_road_scale:
                im = im0s
                if stream:
                    im = im0s[0]

                if self.detect_area:

                    self.detect_area = [self.rescale_detect_area(line, im) for line in self.detect_area ]
                    

                self.road_area = [ [ self.rescale_coords(point,  im) for point in area ] for area in self.road_area ]                
                self.mplt_path = [mpltPath.Path(area) for area in self.road_area]
                self.is_road_scale = True

            if np.array_equal(im0s, old_im0s):
                time.sleep(0.01)
                continue

            old_im0s = im0s


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

                        
                        bbox = np.array([
                        [xyxy[0].item(), xyxy[1].item()],
                        [xyxy[2].item(), xyxy[3].item()] ])



                        norfair_detections.append(Detection(bbox, np.array([conf, cls])))


                tracked_objects = self.tracker.update(detections=norfair_detections)

                if tracked_objects:

                    if times is None:
                        bbox_xyxy, track_data =  self.process_tracking_data(tracked_objects, img, im0,dataset.time_mili)
                    else:

                        bbox_xyxy, track_data =  self.process_tracking_data(tracked_objects, img, im0, times)

                    
                    for obj in track_data:
                        if obj.velocity or (obj.cls in [4 ,5 ,6 ,7 ,9] or obj.person and obj.frame is not None ) or obj.is_stopped:
                            self.send_data(obj, names=names, gn=gn)

                    draw_boxes(im0, bbox_xyxy, track_data)
                    draw_detection_area(im0, self.detect_area)


                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
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


    

    #print(camera.source)
    camera =  Camera()
    # dunas = Camera()

    with torch.no_grad():
        camera.detect(opt)
