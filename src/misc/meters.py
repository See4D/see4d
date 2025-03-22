
from easydict import EasyDict as edict

class AverageMeters(object):
    """Computes and stores the average and current value"""
    def __init__(self, keys=['MV-psnr']):
        self.meters = edict()
        self.keys = keys
        self.reset()

    def reset(self):
        for key in self.keys:
            self.meters[key] = {}
            self.meters[key].val = 0
            self.meters[key].avg = 0
            self.meters[key].sum = 0
            self.meters[key].count = 0

    def reset_by_key(self, key):
        self.meters[key] = {}
        self.meters[key].val = 0
        self.meters[key].avg = 0
        self.meters[key].sum = 0
        self.meters[key].count = 0

    def update(self, in_dict, n=1):
        for key, val in in_dict.items():
            if key not in self.keys:
                self.keys.append(key)
                self.reset_by_key(key)
            self.meters[key].val = val
            self.meters[key].sum += val * n
            self.meters[key].count += n
            self.meters[key].avg = self.meters[key].sum / self.meters[key].count