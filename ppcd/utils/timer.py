import time


class TimeAverager(object):
    def __init__(self):
        self.cnt = 0
        self.batch_time = 0
        self.batch_samples = 0
        self.reset()

    def reset(self):
        self.cnt = 0
        self.batch_time = 0
        self.batch_samples = 0

    def record(self, usetime, num_samples=None):
        self.cnt += 1
        self.batch_time += usetime
        if num_samples:
            self.batch_samples += num_samples

    def get_average(self):
        if self.cnt == 0:
            return 0
        return self.batch_time / float(self.cnt)

    def get_ips_average(self):
        if not self.batch_samples or self.cnt == 0:
            return 0
        return float(self.batch_samples) / self.batch_time


def calculate_eta(remaining_step, speed):
    if remaining_step < 0:
        remaining_step = 0
    remaining_time = int(remaining_step * speed)
    result = "{:0>2}:{:0>2}:{:0>2}"
    arr = []
    for i in range(2, -1, -1):
        arr.append(int(remaining_time / 60**i))
        remaining_time %= 60**i
    return result.format(*arr)