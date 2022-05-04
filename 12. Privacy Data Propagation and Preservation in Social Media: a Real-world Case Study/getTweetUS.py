#!/usr/bin/env python
# coding=utf-8
'''
从原始采到的数据中提取需要使用的推文，
按照转发量来筛选，去重之后按照原始推文分表存储
从推文表中提取用户存到单独的表里
根据用户交集构推文关系图 找最大团确定目标推文
提取用户群和这些用户转发的推文，对用户群采关注者列表
按照传播图构图规则构图
'''

import networkx as nx
from networkx.algorithms.approximation import clique
from dalib.lab.shortcut.mongo import FE2020
from dalib.lab.shortcut.mongo import FE2020_AllTweets
from Mongodb import Mongodb
from dalib.target.twitter.TwitterApi import *
from dalib.lab.settings import *
from concurrent.futures import *
from dalib.common.util import CacheManager
import pymongo

times = {'Oct14', 'Oct15', 'Oct16', 'Oct17', 'Oct18', 'Oct19', 'Oct20', 'Oct21',
         'Oct22', 'Oct23', 'Oct24', 'Oct25', 'Oct26', 'Oct27', 'Oct28', 'Oct29',
         'Oct30', 'Oct31', 'Nov01', 'Nov02', 'Nov03', 'Nov04', 'Nov05', 'Nov06',
         'Nov08', 'Nov09', 'Nov10', 'Nov11', 'Nov12', 'Nov13', 'Nov14', 'Nov15',
         'Nov16', 'Nov17', 'Nov18', 'Nov19', 'Nov20', 'Nov21', 'Nov22', 'Nov23', 'Nov24'}
Trump_tweets_id = ['804', '1008', '902', '897', '1234', '1162', '918', '1252', '845', '811', '809', '939', '1312',
                   '1293', '1210', '989', '1102']


class DBProcesser:
    def __init__(self, **kwargs):
        self.server = Mongodb(kwargs.get('ip') or "121.48.165.123",
                              kwargs.get('port') or 30011,
                              kwargs.get('database') or "FactCheck",
                              kwargs.get('username') or "readAnyDatabase",
                              kwargs.get('password') or "Fzdwxxcl.121")
        max_worker = kwargs.get('max_worker') or 4
        self.thread_pool = ThreadPoolExecutor(max_workers=max_worker, thread_name_prefix="DBProcesser")

        token = TokenDict.twitter.default
        cm = CacheManager(prefix='twitterapi')
        self.api = TwitterApi(cm=cm, proxy=ServerDict.proxy.new, **token)

    # 获取转发量在一定范围内的推文
    def get_some_twitter(self, to_col):
        obj = FE2020()
        to_col = self.server.get_collection(to_col)
        tbs = list(filter(lambda name: name.startswith('Final'), obj.list()))
        tbs.remove('Final')
        tbs = sorted(tbs, key=lambda name: int(name.split('_')[-1]), reverse=True)
        print(tbs)
        tbs = list(filter(lambda name: name.endswith('1122'), tbs))
        for tb in tbs:  # 停在哪个就从哪里开始
            clc = obj.get_collection(tb)
            docs = clc.find({'retweeted_status.retweet_count': {"$gte": 1000}})  # 转发量范围是1000-
            count = 0
            for doc in docs:
                try:
                    to_col.insert_one(doc)
                    count = count + 1
                    if count % 50 == 0:
                        print("从" + tb + "插入了", count, "条推文")
                except:
                    count = count + 1
                    if count % 50 == 0:
                        print("从" + tb + "插入了", count, "条推文")
                    continue

    def get_from_cols(self):
        obj = FE2020()
        all_tbs = obj.list()
        tbs = list(filter(lambda name: name.startswith('Final'), all_tbs))
        tbs.remove('Final')
        tbs = sorted(tbs, key=lambda name: int(name.split('_')[-1]), reverse=True)
        print(tbs)
        return tbs, all_tbs

    # 从固定转发量的推文中提取原始推文 保存在一个新表中
    def get_init_tweet(self, from_col, to_col):
        from_col = self.server.get_collection(from_col)
        to_col = self.server.get_collection(to_col)

        cursors = from_col.find()
        insert_id = set()

        count = 0
        for cursor in cursors:
            if cursor.__contains__('retweeted_status'):
                id = cursor['retweeted_status']['id']
                if id in insert_id:
                    continue
                else:
                    insert_id.add(id)
                    to_col.insert_one(cursor['retweeted_status'])
                    count = count + 1
            else:
                id = cursor['id']
                if id not in insert_id:
                    insert_id.add(cursor['id'])
                    to_col.insert_one(cursor)
                    count = count + 1
            if count % 20 == 0:
                print("已经插入了", count, "条原始推文")

    # 根据日期来控制查询的推文，加快查询速度
    def get_tweets(self, init_tweets):
        """
        获取init_tweets中的所有原始推文的转发推文，并根据原始推文id分表存储，表名为tweet_{id}
        :param init_tweets: 存放原始推文的表名， str，该表中存的每条推文都是原始推文
        :return:
        """
        from_col, all_col_names = dbp.get_from_cols()  # 存储所有推文的表名
        all_init_tweets = dbp.server.get_collection(init_tweets)
        init_tweets = all_init_tweets.find()  # 所有原始推文

        count = 0
        if init_tweets != None:
            for tweet in init_tweets:  # 对每一条原始推文进行转发推文提取
                time = tweet['created_at']
                id = tweet['id']
                time = time.split()
                t = time[1] + time[2]
                if t in times:
                    t = t.replace('Nov', str(11)).replace('Oct', str(10))
                    count = count + 1
                    if count < 0:  # 停在哪就从哪重新开始  默认从0开始
                        continue
                    print("正在操作第", count, "条原始推文")

                    for col_name in from_col:
                        print("正在查询" + col_name + "中的数据")
                        this_time = col_name[-4:]
                        if this_time >= t:  # 分析原始推文的发表时间，在该时间之后进行查询转发推文的操作
                            if 'tweet_'+str(id) not in all_col_names:
                                dbp.get_tweet_by_id(tweet, col_name, 'tweet_' + str(id))
                            else:
                                print("collection tweet_{} exists".format(id))

    # 根据原始推文查询其他的表 返回原始推文对应的所有转发推文
    def get_tweet_by_id(self, init_tweet, from_tweet_col, to_col):
        """
        从from_tweet_col中找到init_tweet的转发推文并存入to_col中
        :param init_tweet: 原始推文，twitterapi返回的推文格式
        :param from_tweet_col:
        :param to_col:
        :return:
        """
        all_tweet_col = self.server.get_collection(from_tweet_col)
        all_tweets = all_tweet_col.find({'retweeted_status': {
            '$exists': True,
        }, "retweeted_status.id": init_tweet['id']})

        to_col = self.server.get_collection(to_col)
        try:
            to_col.insert_one(init_tweet)
        except:

            pass
        # print(all_tweets)
        count = 0
        for cur in all_tweets:
            count = count + 1
            if to_col.find_one({'id': cur['id']}) == None:
                try:
                    to_col.insert_one(cur)
                except:
                    print("insert failed")
                    continue

            if count % 20 == 0:
                print("已经插入了", count, "条转发推文")

    # # 寻找原始推文和所有转发用户
    # for tweet_id in tweet_ids:
    # 	tweets_in_one = all_tweet.find()

    # 对每一个推文表创建对应的用户表
    def insert_users(self):
        obj = FE2020_AllTweets()
        tbs = list(filter(lambda name: name.startswith('tweet'), obj.list()))
        print(tbs)

        count = 0
        for tb in tbs:
            count = count + 1
            if count < 130:  # 停在哪里就从哪里开始
                continue
            from_col = self.server.get_collection(tb)
            documents = from_col.find()  # 一个表里的所有推文  只对其提取用户信息即可

            to_col_name = tb + '_User'
            to_col = self.server.get_collection(to_col_name)

            flag = 0
            for doc in documents:
                if doc.__contains__('user'):
                    screen_name = doc['user']['screen_name']
                else:
                    continue
                if to_col.find_one({'screen_name': screen_name}) == None:  # 表中不存在该用户，则插入·
                    try:
                        to_col.insert_one(doc['user'])
                        flag = flag + 1
                    except:
                        continue
                else:
                    continue
            print("成功创建了" + str(count) + "个用户列表")

    def get_UserGroup(self):
        obj = FE2020_AllTweets()
        tbs = list(filter(lambda name: name.endswith('User'), obj.list()))  # 读取所有的用户表
        # print(tbs)

        all_users = []  # 二维列表 存储所有推文的所有用户
        all_nums = []  # 存储列表对应的推文标号
        max_common = []  # 存储最大的公共节点群
        max_Group = set()  # 存储具有最大公共用户群的表编号

        count = 0
        for tb in tbs:
            count = count + 1
            # if count > 200: break
            num = tb.split('_')[1]  # 编号
            users = []
            user_col = self.server.get_collection(tb)
            cursor = user_col.find()
            for c in cursor:
                users.append(c['screen_name'])
            all_users.append(users)
            all_nums.append(num)
            if len(all_users) % 50 == 0:
                print("已经获取了" + str(len(all_users)) + "个用户列表")
        print("已经获取了" + str(len(all_users)) + "个用户列表")

        # 构建一个以有公共用户作为连边 推文编号为节点的图
        G = nx.Graph()
        for i in range(len(all_users)):
            G.add_node(all_nums[i])
            for j in range(i + 1, len(all_users)):
                if len(set(all_users[i]) & set(all_users[j])) > 100 and len(
                        set(all_users[i]) & set(all_users[j])) <= 500:
                    G.add_edge(all_nums[i], all_nums[j], attr=len(set(all_users[i]) & set(all_users[j])))
        nx.write_gexf(G, "graph3.gexf")

    # 获取公共转发用户图的最大团
    # G = nx.read_gexf("graph3.gexf")
    # # Group = nx.enumerate_all_cliques(G)
    # Group = clique.max_clique(G)
    # print(list(Group))
    # Group_size = len(list(Group))
    #
    # print(Group_size)
    #
    # subgraph = G.subgraph(Group)
    # nx.write_gexf(subgraph, "sub_graph3.gexf")

    # 由多用户表获取其公共部分  参数为存储用户表名的列表和存储公共用户的表名
    def get_common_Userparts(self, tweets_Users_cols, to_col):
        common_User = self.server.get_collection(to_col)
        all_users = []
        for users_col in tweets_Users_cols:
            user_list = []
            users = self.server.get_collection(users_col)
            users = users.find()
            for user in users:
                user_list.append(user['screen_name'])
            all_users.append(user_list)

        common = list(set(all_users[0]).intersection(set(all_users[1]), set(all_users[2]), set(all_users[3]),
                                                     set(all_users[4])))

        # common = list(set(all_users[0]).intersection(set(all_users[1]),set(all_users[2]),set(all_users[3]),
        # 										 set(all_users[4]),set(all_users[5]),set(all_users[6]),
        # 										 set(all_users[7]),set(all_users[8]),set(all_users[9]),
        # 										 set(all_users[10]),set(all_users[11]),set(all_users[12]),
        # 										 set(all_users[13]),set(all_users[14]),set(all_users[15]),set(all_users[16])))

        print(len(common))
        for name in common:
            col = self.server.get_collection(tweets_Users_cols[0])
            cursor = col.find_one({'screen_name': name})
            common_User.insert_one(cursor)

    def get_CommonUser_tweet(self, user_col):
        user_col = self.server.get_collection(user_col)
        users = user_col.find()
        user_list = []
        for col in users:
            user_list.append(col['screen_name'])

        tweets_names = ['tweet_804', 'tweet_1008', 'tweet_902', 'tweet_897', 'tweet_1234']

        # tweets_names = ['tweet_804','tweet_1008','tweet_902','tweet_897','tweet_1234','tweet_1162','tweet_918','tweet_1252','tweet_845',
        #          'tweet_811','tweet_809','tweet_939','tweet_1312','tweet_1293','tweet_1210','tweet_989','tweet_1102']
        count = 0
        for name in tweets_names:
            count = count + 1
            print(count)
            out_name = 'Common_top5_' + name
            col = self.server.get_collection(name)
            to_col = self.server.get_collection(out_name)
            for user in user_list:
                tmp = col.find_one({'user.screen_name': user})
                to_col.insert_one(tmp)


if __name__ == '__main__':
    dbp = DBProcesser(username='dlx', password='kb314dlx', database='FE2020', max_worker=4)
    # dbp1 = DBProcesser(username='dlx', password='kb314dlx', database='FE2020_AllTweets', max_worker=4)   # 处理后的存储每一条单独推文database
    # dbp.get_some_twitter('new_Final1122')         # 这个表存储所有的转发量符合要求的推文
    # dbp.get_init_tweet('new_Final1124', 'origin_tweets_1124')  # 第二个表存储所有原始推文（去重）
    dbp.get_tweets('origin_tweets_1124')							# 获取所有的转发推文
    # dbp1.insert_users()									# 对转发推文提取用户
    # dbp1.get_UserGroup()
    #
    # Trump_tweets_name = []
    # Trump_tweets = []
    # for id in Trump_tweets_id:
    # 	Trump_tweets_name.append('tweet_' + id + '_User')
    # 	Trump_tweets.append('tweet_' + id)
    # dbp1.get_common_Userparts(Trump_tweets_name,'Common_top5_Users')  # 传入两个数组 分别是推文用户的17个表和对应的推文表
    # dbp1.get_CommonUser_tweet('Common_top5_Users')
    # 把图构在一起
