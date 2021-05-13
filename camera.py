import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
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

class Camera:


    def __init__(self):
        self.source = ""
        self.road_area = [(0,0), (0,0), (0,0) ,(0,0)]
        self.mplt_path = mpltPath.Path(self.road_area)


    
    @property
    def palette(self):
        """ bounding boxes colors """
        return (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    # check if object is inside road
    def inside_road(self, x, y):
        """ check if object is inside road or not """

        return self.mplt_path.contains_point((x, y), radius=1e-9)


    def box_center(self, *xyxy):
        """ calculate bbox center based xy points """

        #bbox = xyxy[0]
        x1, y1, x2, y2 = xyxy

        return (int(x1) + int(x2))/2 , (int(y1) + int(y2)) /2

    def isMotocycle(self, center_x, center_y):
        """ check if detected object is an motocycle

            if ouside of road - cyclist
            else - motocycle
        """

        if (self.inside_road(center_x, center_y)):
            return True
        
        return False
        

    def detect(self, opt, save_img=False):

        out, weights, view_img, imgsz = \
            opt.output, opt.weights, opt.view_img, opt.img_size


        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        device = select_device(opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model yolo5
        model = torch.load(weights, map_location=device)[
            'model'].float()  # load to FP32
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        # Definir o tipo de entrada ( video / camera)
        vid_path, vid_writer = None, None

        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(self.source , img_size=imgsz)


        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        names.append('motocycle')
        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        # run once
        _ = model(img.half() if half else img) if device.type != 'cpu' else None

        save_path = str(Path(out))
        txt_path = str(Path(out)) + '/results.txt'

        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(
                pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            print("pred : ", pred)
            # Process detections
            for i, det in enumerate(pred):  # detections per image

                # det - pytorch tensor (matriz)

                print("det: ", det)
                print(det[:, -1])

                # det- detections

                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()


                s += '%gx%g ' % img.shape[2:]  # print string
                save_path = str(Path(out) / Path(p).name)

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        print("c: ", c)
                        print("name: ", names[int(c)])
                        # c - classe
                        # n - numero de objetos detetados para a classe
                        # names - dicionario com o nome das classes
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    bbox_xywh = []
                    confs = []
                    classes = []
                    print(det[:, -1])

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        # cls - classe
                        x_c, y_c, bbox_w, bbox_h = self.bbox_rel(*xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                        print("cls : ", cls)

                        classes.append(cls)

                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    print("antes: ", bbox_xywh)
                    print("depois: ", xywhs)

                    # im0 - imagem

                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, im0, classes)
                    print("outputs: ", outputs)
                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -2]
                        classes = outputs[:, -1]

                        self.draw_boxes(im0, bbox_xyxy, [names[int(c)] for c in classes], identities)

                    # Write MOT compliant results to file
                    if len(outputs) != 0:
                        for j, output in enumerate(outputs):

                            json_data = self.send_data(output, names)

                            print(json_data)

                else:
                    deepsort.increment_ages()

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    print('saving img!')
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)
                    else:
                        print('saving video!')
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer

                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            vid_writer = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                        vid_writer.write(im0)


        print('Done. (%.3fs)' % (time.time() - t0))


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


    def send_data(self, output, names):
        bbox_left = output[0]
        bbox_top = output[1]
        bbox_w = output[2]
        bbox_h = output[3]
        # id
        identity = output[-2]
        c = output[-1]

        center_x, center_y = self.box_center(output[0 :4])
        # bike
        if c == 1 and self.isMotocycle(center_x, center_y):

            c = len(names) - 1

        return json.dumps({"classe": names[int(c)],
                           "box_left": int(bbox_left), "box_top": int(bbox_top),
                           "box_w": int(bbox_w), "box_h": int(bbox_h), "inside": self.inside_road(center_x, center_y)})
