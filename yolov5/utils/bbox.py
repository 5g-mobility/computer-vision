
import cv2



def box_center(xyxy):
    """ calculate bbox center based xy points """

    x1, y1, x2, y2 = xyxy

    return (int(x1) + int(x2)) / 2, (int(y1) + int(y2)) / 2


def palette():
    """ bounding boxes colors """
    return (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)



def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette()]
    return tuple(color)


def draw_boxes(img, bbox, track_objects, offset=(0, 0)):
    
    """ draw bbox of each object in frame """ 

    for i, box in enumerate(bbox):

        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = track_objects[i].idx
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img


def resize(img, img_size):

    " resize image to x, y dimensions"

    return cv2.resize(img, img_size)


def draw_detection_area(img, detect_area):

    for line in detect_area:

        p1, p2 = line
        cv2.line(img, p1, p2, (0,0,255),2)

    return img



def is_inside_area(point, line1, line2 ):

    l1p1 , l1p2 = line1

    l2p1 , l2p2 = line2
    


    l1v1 = (l1p2[0] - l1p1[0], l1p2[1] - l1p1[1])

    l1v2 = (l1p2[0] - point[0], l1p2[1] - point[1])

    l2v1 = (l2p2[0] - l2p1[0], l2p2[1] - l2p1[1])

    l2v2 = (l2p2[0] - point[0], l2p2[1] - point[1])

    exp1 = l1v1[0]*l1v2[1] - l1v1[1]* l1v2[0]

    exp2 = l2v1[0]*l2v2[1] - l2v1[1]* l2v2[0]



    return exp1<0 and exp2 > 0
