# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 16:57
# @Author  : whm
# @Email   : whm0602@qq.com
# @File    : threadpool.py
# @Software: PyCharm

from concurrent.futures import *
import threading
import time

def test(string):
    print("thread {} is running, arg is '{}'".format(threading.current_thread().name, string))
    time.sleep(3)
    return 1

if __name__ == '__main__':
    thread_pool = ThreadPoolExecutor(max_workers=10, thread_name_prefix="test_")
    for i in range(0,20):
        future = thread_pool.submit(test, "hello, {}".format(i))

    thread_pool.shutdown(wait=True)