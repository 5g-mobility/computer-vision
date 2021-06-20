

class DataObject:

    def __init__(self, idx, xyxy, cls, conf, is_stopped,  velocity = None, frame = None, firstTack= None ):
        self.idx = idx
        self.xyxy= xyxy
        self.cls = cls
        self.conf = conf
        self.is_stopped = is_stopped
        self.velocity =  velocity
        self.frame = frame
        self.firstTack = firstTack