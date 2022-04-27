# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 15:02
# @Author  : whm
# @Email   : whm0602@qq.com
# @File    : RetweetAcquisition.py
# @Software: PyCharm

from Mongodb import Mongodb
from dalib.target.twitter.TwitterApi import *
from dalib.lab.settings import *
from time import sleep


def get_retweets_of(init_id, retweet_col_name, rewteet_users_col_name):
    """
    获取某个原始推文id的最近100条转推推文（twitterapi限制只能一百条），以及转发给推文的用户信息
    :param init_id:原始推文id
    :param retweet_col_name:获取的信息的存放位置
    :return:
    """
    token = TokenDict.twitter.account2
    api = TwitterApi(proxy=ServerDict.proxy.default, **token)
    # api = TwitterApi(proxy=None, **token)
    api.withCacheManager('twitterapi')

    server = Mongodb("121.48.165.123", 30011, "FactCheck", "dlx", "kb314dlx")
    col = server.get_collection(retweet_col_name)
    user_col = server.get_collection(rewteet_users_col_name)

    # 获取原始推文
    if col.find_one({'id': init_id}) == None:
        result = api.getTweetById(init_id)
        col.insert_one(result)
    else:
        print("id {} alraedy exists".format(init_id))

    # 获取100条转推推文
    cursor = api.getRetweetsByTweetId(init_id)
    for document in cursor:
        if document.__contains__('errors'):
            print("errors: {}".format(document['errors']))
            continue

        if col.find_one({'id': document['id']}) == None:
            col.insert_one(document)
        else:
            print("id {} alraedy exists".format(document['id']))


    # 获取retweet_users
    retweet_users_col = server.get_collection(rewteet_users_col_name)
    cursor = -1
    while cursor != 0:
        result = api.getRetweetersIds(id=init_id, cursor=cursor)
        for id in result['ids']:
            if retweet_users_col.find_one({'id': id}) != None:
                print("user {} already exists".format((id)))
                continue
            user = api.getUser(user_id=id)
            if user.__contains__('errors'):
                print('getUser failed, error: {}'.format(user['errors']))
                continue

            if retweet_users_col.find_one({'id': id}) != None:
                print("user {} already exist".format(id))
            retweet_users_col.insert_one(user)
            print("insert user {} successfully".format((user)))

        cursor = result['next_cursor']


def get_user(user_id, api):
    user = api.getUser(user_id=user_id)
    if user.__contains__('errors'):
        print('getUser failed, error: {}'.format(user['errors']))
        return None
    followers_ids = api.getFollowersIds(user_id=user_id)
    while followers_ids.__contains__('errors'):
        print('Rate limit exceeded, retry in 30 seconds')
        sleep(30)
        print('retry...')
        followers_ids = api.getFollowersIds(user_id=id)
    user['followers_ids'] = followers_ids['ids']
    sleep(5)
    return user


if __name__ == '__main__':
    get_retweets_of(1234924508500447234, "Retweet2", "tmp")