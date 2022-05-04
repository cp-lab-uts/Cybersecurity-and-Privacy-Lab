# -*- coding: utf-8 -*-
# @Time    : 2020/5/28 17:08
# @Author  : whm
# @Email   : whm0602@qq.com
# @File    : DataAcqUnit.py
# @Software: PyCharm

from concurrent.futures import *
import threading
import time
from dalib.lab.settings import TokenDict
from TwitterSelenium import TwitterSelenium
from DBProcesser import DBProcesser

class DataAcqUnit:
    def __init__(self, **kwargs):
        max_worker = kwargs.get('max_worker') or 4
        self.thread_pool = ThreadPoolExecutor(max_workers=max_worker, thread_name_prefix="DataAcqUnit")
        self.dbp = DBProcesser(username='dlx', password='kb314dlx')

    def getfollowersname(self, name):
        self.thread_pool.submit(self._getfollowersname, name)

    def _getfollowersname(self, name):
        obj = TwitterSelenium()
        obj.login(**TokenDict.twitter.account)
        screen_names = obj.getFollowersName(name)
        print(screen_names)
        #obj.quit()
        self.dbp.insert_followers_screen_name({'screen_name': name,
                                               'followers_screen_names': list(screen_names)},
                                                "tmp")
        print("!!!!!")

    # def __del__(self):
    #     self.thread_pool.shutdown(wait=True)
    def insert(self):
        self.dbp.insert_followers_screen_name({'screen_name': 'ssas', 'followers_screen_names':list( {'sssss','aaa'})},
                                              "tmp")

if __name__ == '__main__':
    daqu = DataAcqUnit()
    daqu.getfollowersname('DrREpstein')
    # daqu.getfollowersname('LuvMalibuBarbie')
    # daqu.getfollowersname('kindlee55')
    # daqu.getfollowersname('DrREpstein')
    # daqu.insert()