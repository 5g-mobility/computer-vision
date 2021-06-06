

class DataObject:

    def __init__(self, idx, xyxy, cls, conf, n_stop,  velocity = None, frame = None ):
        self.idx = idx
        self.xyxy= xyxy
        self.cls = cls
        self.conf = conf
        self.n_stop = n_stop
        self.velocity =  velocity
        self.frame = frame