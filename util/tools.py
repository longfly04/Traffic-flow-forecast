import time
import datetime as dt
import sys,os
import logging
from functools import wraps
import traceback

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
        print('[Timer] Time taken: %s' % (end_dt - self.start_dt))

class Logger():
    def __init__(self,):
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),"logs")
        today = time.strftime('%Y%m%d', time.localtime(time.time()))
        full_path = os.path.join(log_dir, today)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
        self.log_path = os.path.join(full_path,"traffic_flow_forecast.log")

    def get_logger(self, ):
     # 获取logger实例，如果参数为空则返回root logger
        logger = logging.getLogger("facebook")
        if not logger.handlers:
            # 指定logger输出格式
            formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')
 
            # 文件日志
            file_handler = logging.FileHandler(self.log_path, encoding="utf8")
            file_handler.setFormatter(formatter)  # 可以通过setFormatter指定输出格式
 
            # 控制台日志
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.formatter = formatter  # 也可以直接给formatter赋值
 
            # 为logger添加的日志处理器
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
 
            # 指定日志的最低输出级别，默认为WARN级别
            logger.setLevel(logging.INFO)
     #  添加下面一句，在记录日志之后移除句柄
        return  logger


def info(func):
    @wraps(func)
    def log(*args,**kwargs):
        logger = Logger()
        try:
            print("[INFO] Function: \" {name} \" is starting...".format(name = func.__name__))
            timer = Timer()
            timer.start()
            result = func(*args,**kwargs)
            timer.stop()
            print("[INFO] Function: \" {name} \" is completed .".format(name = func.__name__))
            return result
        except Exception as e:
            logger.get_logger().error(f"{func.__name__} is error,here are details:{traceback.format_exc()}")
    return log


