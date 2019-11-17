import time
import datetime as dt
import sys,os

class Timer():
    '''
    定义一个计时器类 stop方法输出用时
    '''

    def __init__(self):
        self.start_dt = None

    def start(self):
        self.start_dt = dt.datetime.now()

    def stop(self):
        end_dt = dt.datetime.now()
        print('Time taken: %s' % (end_dt - self.start_dt))



