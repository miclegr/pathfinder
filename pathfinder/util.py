import numpy as np


class StackedTensorTransformer():
    def __init__(self, event_shapes):
        self.shapes = [s[0] if len(s) > 0 else 1 for s in event_shapes]
        self.shapes_cumsum = [sum(self.shapes[0:i])
                              for i in range(len(self.shapes)+1)]

    def stack(self, list_of_arrs):
        return np.hstack([list_of_arrs[i] if list_of_arrs[i].ndim >= 2
                          else list_of_arrs[i][:, None]
                          for i in range(len(self.shapes))])

    def unstack_pure_fn(self):
        def fn(t):
            out = [t[i:j] if t.ndim == 1 else t[:, i:j]
                   for i, j in zip(self.shapes_cumsum,
                                   self.shapes_cumsum[1:])]

            return [t[:, 0] if t.ndim == 2 and t.shape[1] == 1
                    else t for t in out]
        return fn
