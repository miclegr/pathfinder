import numpy as np


class StackedTensorTransformer():
    def __init__(self, event_shapes):
        self.shapes = [s[0] if len(s) > 0 else 1 for s in event_shapes]
        self.shapes_cumsum = [sum(self.shapes[0:i]) for i in range(len(self.shapes)+1)] 
    def stack(self, l):
        return np.hstack([l[i] for i in range(len(self.shapes))])
    def unstack(self, t):
        return [t[i:j] for i,j in zip(self.shapes_cumsum,self.shapes_cumsum[1:])]
