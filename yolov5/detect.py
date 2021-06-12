import argparse
import re
import time
from pathlib import Path
import os

from torch.functional import Tensor
from tracker import Detection, Tracker
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import math
import json

import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import matplotlib.path as mpltPath


def resize(img, x, y):

    return cv2.resize(img, (x, y))

def isMotocycle(path, center_x, center_y):
    """ check if detected object is an motocycle

        if ouside of road - cyclist
        else - motocycle
    """

    if (inside_road(path, center_x, center_y)):
        return True

    return False

def send_data(*xyxy, c, names, path):
    
    bbox_left = xyxy[0]
    bbox_top = xyxy[1]
    bbox_w = xyxy[2]
    bbox_h = xyxy[3]
    # id


    center_x, center_y = box_center(xyxy)
    # bike
    if c == 1 and isMotocycle(path, center_x, center_y):

        c = len(names) - 1

    return json.dumps({"classe": names[int(c)],
                        "box_left": int(bbox_left), "box_top": int(bbox_top),
                        "box_w": int(bbox_w), "box_h": int(bbox_h), "inside": inside_road(path, center_x, center_y)})



palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

max_distance_between_points = 30

def euclidean_distance(detection, tracked_object):
    return np.linalg.norm(detection.points - tracked_object.estimate)


def estimateSpeed(location1, location2, fps):
    
    print(location1)
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
	# ppm = location2[2] / carWidht
    # real = 100
    # 599.923537
    
    ppm = 10
    d_meters = d_pixels / ppm
	#print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
	#fps = 18
    speed = d_meters * fps * 3.6
    return speed
	

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def bbox_rel(*xyxy):
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

def draw_boxes(img, track_objects, identities, offset=(0, 0)):
    for i, box in enumerate(track_objects):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img



def inside_road(path, center_x, center_y):
    #top_left, top_right, bottom_left, bottom_right

    return path.contains_point((center_x, center_y), radius=1e-9)


def box_center(*xyxy):
    print(xyxy)
    bbox = xyxy[0]
    x1, y1, x2, y2 = bbox
    return (int(x1) + int(x2))/2 , (int(y1) + int(y2)) /2

    
def detect(pid, save_img=False):
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    view_img =  True
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))



    mplt_path = mpltPath.Path([(828, 287),(1345, 1296),(2143, 1296),(960, 287)])
    
    img_size = (2304,1296) # x, y

    tracker = Tracker(
        distance_function=euclidean_distance,
        distance_threshold=max_distance_between_points,
    )



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


    print("image_size: ", img_size)
    # TODO: usar o recise na imagem 


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
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)


    fps =  dataset.fps
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    names.append('motocycle')
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    print(dataset.fps)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()


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
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]] # normalization gain whwh
            
            
            
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                #print(det[:, :4])
            
                

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                #bbox_xywh = []
                #confs = []
               
                

                # Write results
                for *xyxy, conf, cls in reversed(det):

                    center_x, center_y = box_center(xyxy)
                    

                    ret = inside_road(mplt_path, center_x, center_y)
                    print(ret)
                    if save_txt:  # Write to file
                        
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                        
                        #print( inside_road( ))
                        line = (cls, *xyxy, conf,ret ) if opt.save_conf else (cls, center_x, center_y, ret)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        print(send_data(*xyxy,c= cls, names= names,path= mplt_path))

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                
                    bbox = np.array([
                        [xyxy[0].item(), xyxy[1].item()],
                        [xyxy[2].item(), xyxy[3].item()]
                ])

                

                    norfair_detections.append(Detection(bbox, np.array([conf, cls])))
                    #x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    #obj = [x_c, y_c, bbox_w, bbox_h]
                    #bbox_xywh.append(obj)
                    #confs.append([conf.item()])
                    

                #xywhs = torch.Tensor(bbox_xywh)
                #confss = torch.Tensor(confs)

                    # if save_txt :  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                    #     with open(txt_path + '.txt', 'a') as f:
                    #         f.write(('%g ' * len(line)).rstrip() % line + '\n')


                # if save_img or view_img:  # Add bbox to image
                #     label = f'{names[int(cls)]} {conf:.2f}'
                #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)


            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

        print(len(norfair_detections))
        tracked_objects = tracker.update(detections=norfair_detections)
        print("######", len(tracked_objects))
        bbox = []
        bbox_xyxy = []
        bbox_last = []
        if len(tracked_objects) > 0:
            
            
            
            for obj in tracked_objects:

                if obj.previous_detection:
                    bbox_last.append([y  for b in obj.previous_detection.points.tolist() for y in b])

                for box in obj.last_detection.points.tolist():
                    for x in box:
                        bbox.append(x)

                bbox_xyxy.append(bbox)
                bbox = []
            #print(torch.tensor(bbox_xyxy))
            bbox_xyxy = scale_coords(
            img.shape[2:], torch.tensor(bbox_xyxy), im0.shape).round()

     
            # objects_detection = [obj.last_detection.points.tolist() for obj in tracked_objects]
            # bbox_xyxy = [obj for x in objects_detection.points for obj in x ]
            identities = [obj.id for obj in tracked_objects]

            bbox_xywh = xyxy2xywh(bbox_xyxy)
            #print(bbox_last)


            if bbox_last != []:


                bbox_last = scale_coords( img.shape[2:], torch.tensor(bbox_last), im0.shape).round()
                velocities = [estimateSpeed(bbox_last[i], bbox_xyxy[i], fps) for i,obj in enumerate(tracked_objects) if obj.previous_detection]

                print(velocities)
            #print(identities)
            draw_boxes(im0, bbox_xyxy, identities)

            print("anda cao")
            # Stream results
            if view_img:
                print("--------------------- view-img")
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
        # except TimeoutError as time_err:
        #     print(f'Timeout error: {time_err}')
        #     flag_repeat_LoadStream = True
        #     dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        # except Exception as e:
        #     print(f'Exception occur when detecting: {e}')

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
    check_requirements(exclude=('pycocotools', 'thop'))
    pid=os.getpid()
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:

            detect(pid)
