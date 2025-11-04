import time
import datetime


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Timer:
    def __init__(self):
        self.t0 = 0
        self.t1 = 0
        self.duration = self.get_duration()

    def start(self):
        self.t0 = time.time()

    def stop(self):
        self.t1 = time.time()
        self.duration = self.get_duration()
        self.fps = int(1 / self.duration)
        return self.duration

    def get_duration(self):
        """duration in seconds"""
        return self.t1 - self.t0

    def print_duration(self, msg="[Elapsed time]", displayms=True, displayfps=True):

        if displayms:
            msg = msg + " {}ms".format(int(self.duration * 1000))

        # if displayfps:
        #     msg = msg + " / {}fps".format(self.fps)

        print(msg)
