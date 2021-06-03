

class DataObject:

    def __init__(self, idx, xyxy, cls, conf, velocity=None ):
        self.idx = idx
        self.xyxy= xyxy
        self.cls = cls
        self.conf = conf
        self.velocity =  velocity