import numpy as np

class MovingAverage():

    def __init__(self, data):
        self.data = data
        self.full_avg = np.average(self.data)

    def get_last_window(self, window_len):
        if self.data.shape[0] >= window_len:
            return self.data[-window_len:]
        else:
            return np.ones(window_len) * self.full_avg