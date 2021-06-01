from re import X
import torch
import torchvision
import numpy as np
import csv
import argparse

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def center2xyxy(x, y):
    
    pass

def main(*xyxy,coords, cls):
    cn = torch.tensor((1296, 2304, 3))[[1, 0, 1, 0]]
    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / cn).view(-1).tolist()
    line = [cls, *xywh, 0.6]
    with open('data.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(line)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--class',choices=['person', 'car', 'truck', 'moto'] ,  type=str, help='source', required=True)
    parser.add_argument('--position',nargs='+',  type=int, help='source', required=True)
    parser.add_argument('--coords',nargs='+',  type=str, help='source', required=True)
    args = parser.parse_args()

    assert len(args.position) == 2
    assert len(args.coords) == 2

    main(args.position, coords = args.coords, cls = 0)