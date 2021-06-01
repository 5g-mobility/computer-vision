

class DataObject:

    def __init__(self, xyxy, cls, conf, velocity=None ) :
        self.xyxy= xyxy
        self.cls = cls
        self.conf = conf
        self.velocity =  velocity