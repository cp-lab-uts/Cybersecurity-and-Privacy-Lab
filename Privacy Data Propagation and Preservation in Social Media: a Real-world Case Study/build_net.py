# -*- coding: utf-8 -*-
# @Time    : 2020/5/27 9:19
# @Author  : whm
# @Email   : whm0602@qq.com
# @File    : build_net.py
# @Software: PyCharm

from Mongodb import Mongodb
from collections import defaultdict
import time
import networkx as nx
import numpy as np
import pickle


def build_nets():
    print('load pickle file...')
    with open('./pickle/user_followers', 'rb') as f:
        all_user_followers_dict = pickle.load(f)
    print('load pickle file successfully')
    tweet_cols = ['tweet_1327425499472293888',
                  'tweet_1327940661258162176',
                  'tweet_1327971951852343297',
                  'tweet_1328402814800977920',
                  'tweet_1328452034450907138',
                  'tweet_1328493390472757253',
                  'tweet_1329107891920527362']

    for col in tweet_cols:
        build_net_by_id(col, 'multi_privacy', 'FE2020_AllTweets', all_user_followers_dict)


def build_net_by_id(tweet_col, user_col, database, followers_count):
    """
    构建转发关系网络, 确保数据中有且只有一条origin_tweet
    从tweet_col中获取对应推文user，从user_col中获取user的关注列表
    :param tweet_col:
    :param user_col:
    :return:
    """
    server = Mongodb("121.48.165.123", 30011, database, "readAnyDatabase", "Fzdwxxcl.121")
    relevant_tweet_col = server.get_collection(tweet_col)
    relevant_user_col = server.get_collection(user_col)

    origin_tweet = None  # (created_at, origin_tweet_id)
    tweet_timestamp = {}  # key: tweet_id, value: timestamp
    retweets = []  # elements: (created_at, retweet_id)
    user_id_followers_dict = defaultdict(list)  # key: user_id, value : [ a list of id's followers ]
    tweet_id_screen_name_dict = {}  # key : tweet_id,  value : screen_name
    user_id_followers_count_dict = {}  # key : user_id, value : followers_count
    tweet_id_user_id_dict = {}

    cursor = relevant_tweet_col.find()

    # print('load pickle file...')
    # with open('./pickle/user_followers', 'rb') as f:
    #     all_user_followers_dict = pickle.load(f)
    # print('load pickle file successfully')
    all_user_followers_dict = followers_count
    # 初始化 tweet_user_dict  origin_twitte_id  origin_retweet_dict
    count = 0
    for document in cursor:
        # 过滤掉有错的推文
        count += 1
        print(f'tweet {count}')
        if document.__contains__('errors'):
            continue
        screen_name = document['user']['screen_name']  # 推文所属用户screen_name
        user = relevant_user_col.find_one({'screen_name': screen_name})
        user_id = user.get('id')
        if user_id is None:
            continue

        tweet_id = document['id']  # 推文id
        tweet_id_screen_name_dict[tweet_id] = screen_name

        created_at = document.get('created_at')  # 推文创建的时间戳
        tweet_timestamp[tweet_id] = created_at

        tweet_id_user_id_dict[tweet_id] = user_id

        followers_ids = all_user_followers_dict.get(user_id)
        if followers_ids is None:
            followers_ids = set()
            print('followers_ids is None')
        user_id_followers_dict[user_id] = followers_ids

        followers_count = user.get('followers_count')
        if followers_count is None:
            followers_count = len(followers_ids)
        user_id_followers_count_dict[user_id] = followers_count

        if not document.__contains__('retweeted_status'):
            print('origin tweet')
            origin_tweet = (created_at, tweet_id)
        else:
            retweets.append((created_at, tweet_id))

    retweets.sort(reverse=True)  # 将转推的推文列表按时间排序

    # build net
    G = nx.DiGraph()
    tmp_list = []  # 存放的是推文id
    index = 0
    origin_created_at, origin_tweet_id = origin_tweet
    origin_screen_name = tweet_id_screen_name_dict[origin_tweet_id]
    if not G.has_node(origin_screen_name):
        G.add_node(origin_screen_name, timestamp=str(origin_created_at),
                   tweet_id=str(origin_tweet_id), is_origin=True, index=index)
        index += 1

    tmp_list.append(origin_tweet_id)

    # 将所有转发者加入图中
    for timestamp, retweet_id in retweets:  # 遍历 原始推文
        retweet_user_screen_name = tweet_id_screen_name_dict[retweet_id]  # 转发者的screen_name
        retweet_user_id = tweet_id_user_id_dict[retweet_id]
        if not G.has_node(retweet_user_screen_name):  # 添加转发者节点
            G.add_node(retweet_user_screen_name,
                       timestamp=str(timestamp), tweet_id=str(retweet_id),
                       origin=str(origin_tweet_id), is_origin=False, index=index)
            index += 1

        flag = False  # 记录转发者是否是前面某个用户的关注者
        if len(tmp_list) == 0:
            flag = True

        # for j in range(len(tmp_list) - 1, -1, -1):
        for j, _ in enumerate(tmp_list):
            j_tweet_id = tmp_list[j]
            j_user_id = tweet_id_user_id_dict[j_tweet_id]
            if retweet_user_id in user_id_followers_dict[j_user_id]:  # 如果转发者是前面某个用户的关注者
                j_user_screen_name = tweet_id_screen_name_dict[j_tweet_id]
                if not G.has_edge(j_user_screen_name, retweet_user_screen_name):
                    G.add_edge(j_user_screen_name, retweet_user_screen_name, weight=1)
                flag = True
                break

        # 如果没有人的关注者列表里有它，就连根据followers数量按概率连接
        if not flag:
            count_list = [user_id_followers_count_dict[tweet_id_user_id_dict[tmp_tweet_id]] for tmp_tweet_id in
                          tmp_list]
            norm_const = sum(count_list)
            if norm_const == 0:
                normalized_probs = [1.0 / len(count_list) for _ in count_list]
            else:
                normalized_probs = [float(count) / norm_const for count in count_list]

            J, q = alias_setup(normalized_probs)
            user_index = alias_draw(J, q)
            tmp_tweet_id = tmp_list[user_index]
            tmp_screen_name = tweet_id_screen_name_dict[tmp_tweet_id]
            if not G.has_edge(tmp_screen_name, retweet_user_screen_name):
                G.add_edge(tmp_screen_name, retweet_user_screen_name, weight=1)

        tmp_list.append(retweet_id)

    nx.write_gexf(G, "./tmp/{}_retweet_net.gexf".format(tweet_col))


def build_net_by_screen_name(tweet_col, user_col, database, process=True):
    """
    构建转发关系网络
    从tweet_col中获取对应推文user，从user_col中获取user的关注列表
    :param tweet_col:
    :param user_col:
    :return:
    """
    server = Mongodb("121.48.165.123", 30011, database, "readAnyDatabase", "Fzdwxxcl.121")
    relevant_tweet_col = server.get_collection(tweet_col)

    origin_twitte_id = set()  # set of origin_tweets_id
    tweet_user_dict = {}  # key: tweet id, value: user screen_name
    # user_tweet_dict = {}  # key: user screen_name, value: tweet id
    tweet_timestamp = {}  # key: tweet id, value: timestamp
    origin_retweet_dict = defaultdict(list)  # key: origin_tweet_id, vlaue: (created_at, retweet_id)
    quote_quoted_dict = {}

    cursor = relevant_tweet_col.find()
    # 初始化 tweet_user_dict  origin_twitte_id  origin_retweet_dict
    for document in cursor:
        # 过滤掉有错的推文
        if document.__contains__('errors'):
            continue

        id = document.get('id')
        if document.__contains__('user'):
            tweet_user_dict[id] = document.get('user').get('screen_name')
            # user_tweet_dict[document.get('user').get('screen_name')] = id

        created_at = document.get('created_at')  # 字符串时间，Fri Mar 13 20:18:44 +0000 2020
        if created_at == None:
            created_at_timestamp = None
        else:
            created_at_timestamp = time.mktime(time.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y"))  # 将字符串时间转换为时间戳

        tweet_timestamp[id] = created_at_timestamp

        if document.__contains__('retweeted_status'):
            tmp_id = document.get('retweeted_status').get('id')  # 原始推文id
            origin_twitte_id.add(tmp_id)
            origin_retweet_dict[tmp_id].append((created_at_timestamp, id))

        if document.__contains__('quoted_status'):
            quote = document['user']['screen_name']
            quoted = document['quoted_status']['user']['screen_name']
            quote_quoted_dict[quote] = quoted

    # print(origin_twitte_id)

    for key in origin_retweet_dict:
        origin_retweet_dict[key].sort()  # 将转推的推文列表按时间排序

    relevant_user_col = server.get_collection(user_col)

    # user_followers_dict = defaultdict(list)  # key:user_id, value: list of follower_id
    user_followers_dict = defaultdict(list)  # key:user_id, value: list of follower_screen_name
    user_followers_count_dict = defaultdict(int)  # key: user screen_name, value: followers_count
    user_cursor = relevant_user_col.find()
    # 初始化screen_namer_followers_dict, user_followers_count_dict
    for document in user_cursor:
        screen_name = document.get('screen_name')
        if document.__contains__('followers_screen_names'):
            user_followers_dict[screen_name] = document['followers_screen_names']
        if document.__contains__('followers_count'):
            user_followers_count_dict[screen_name] = document['followers_count']
        else:
            user_followers_count_dict[screen_name] = 0

    # 按followers_count由大到小对user_followers_count_dict排序
    followers_count_sorted_list = sorted(user_followers_count_dict.items(), key=lambda x: x[1], reverse=True)

    # build net
    G = nx.DiGraph()

    for origin in origin_retweet_dict:
        tmp_list = []
        if tweet_user_dict.__contains__(origin):
            if not G.has_node(tweet_user_dict[origin]):
                G.add_node(tweet_user_dict[origin], timestamp=str(tweet_timestamp[origin]),
                           tweet_id=str(origin), is_origin=True, is_quoted=False, is_quote=False)
            tmp_list.append(tweet_user_dict[origin])

        # 将所有转发者加入图中
        for timestamp, retweet_id in origin_retweet_dict[origin]:  # 遍历 原始推文
            retweet_user_screen_name = tweet_user_dict[retweet_id]  # 转发者的screen_name
            if not G.has_node(retweet_user_screen_name):  # 添加转发者节点
                G.add_node(retweet_user_screen_name,
                           timestamp=str(timestamp), tweet_id=str(retweet_id),
                           origin=str(origin), is_origin=False, is_quoted=False, is_quote=False)

            flag = False  # 记录转发者是否是前面某个用户的关注者
            if len(tmp_list) == 0:
                flag = True

            # for user in tmp_list:
            for j in range(len(tmp_list) - 1, -1, -1):
                user = tmp_list[j]
                if retweet_user_screen_name in user_followers_dict[user]:  # 如果转发者是前面某个用户的关注者
                    if G.has_edge(user, retweet_user_screen_name):
                        G.get_edge_data(user, retweet_user_screen_name)['weight'] += 1
                    else:
                        G.add_edge(user, retweet_user_screen_name, weight=1)
                    flag = True
                    break

            # 如果没有人的关注着列表里有它，就连在followers数最多的user上
            # if not flag:
            #     for user,count in followers_count_sorted_list:
            #         if user in tmp_list:
            #             G.add_edge(user, retweet_user_screen_name)
            #             break

            # 如果没有人的关注者列表里有它，就连根据followers数量按概率连接
            if not flag:
                count_list = [user_followers_count_dict[user] for user in tmp_list]
                norm_const = sum(count_list)
                if norm_const == 0:
                    normalized_probs = [1.0 / len(count_list) for count in count_list]
                else:
                    normalized_probs = [float(count) / norm_const for count in count_list]

                J, q = alias_setup(normalized_probs)
                user_index = alias_draw(J, q)
                user = tmp_list[user_index]
                if G.has_edge(user, retweet_user_screen_name):
                    G.get_edge_data(user, retweet_user_screen_name)['weight'] += 1
                else:
                    G.add_edge(user, retweet_user_screen_name, weight=1)

            tmp_list.append(retweet_user_screen_name)

    # for quote in quote_quoted_dict:
    #     quoted = quote_quoted_dict[quote]
    #     if quote == quoted:
    #         continue
    #     if G.has_node(quoted):
    #         if not G.has_node(quote):
    #             G.add_node(quote, timestamp=str(tweet_timestamp[user_tweet_dict[quote]]),
    #                        tweet_id=str(user_tweet_dict[quote]), is_origin=False, is_quote=True, is_quoted=False)
    #         G.nodes[quote]['is_quote'] = True
    #         G.nodes[quote]['is_quoted'] = True
    #         in_edges = list(G.in_edges(quote))
    #         G.remove_edges_from(in_edges)
    #         G.add_edge(quoted,quote)

    if process:
        # 处理小的弱联通分支
        while (True):
            tmp_c = sorted(nx.weakly_connected_components(G), key=len, reverse=True)[1:]
            if len(tmp_c[0]) == 1:
                break
            for c in tmp_c:
                # print(list(c))
                # print(G.in_degree(list(c)))
                if len(c) == 1:
                    continue
                for node in c:
                    # print(type(node))
                    # print(G.in_degree(node))
                    if G.in_degree(node) == 0:
                        origin = int(G.nodes[node]['origin'])
                        id = int(G.nodes[node]['tweet_id'])
                        pre_retweet = None
                        for timestamp, retweet_id in origin_retweet_dict[origin]:
                            if id == retweet_id:
                                break
                            else:
                                pre_retweet = retweet_id

                        if pre_retweet == None:
                            last_node = tweet_user_dict[origin]
                        else:
                            last_node = tweet_user_dict[pre_retweet]

                        G.add_edge(last_node, node)

    nx.write_gexf(G, "./tmp/retweet_net_{}.gexf".format(int(time.time())))


def build_net_by_screen_name_split(tweet_col, user_col, database):
    """
    构建转发关系网络
    从tweet_col中获取对应推文user，从user_col中获取user的关注列表
    :param tweet_col:
    :param user_col:
    :return:
    """
    server = Mongodb("121.48.165.123", 30011, database, "readAnyDatabase", "Fzdwxxcl.121")
    relevant_tweet_col = server.get_collection(tweet_col)

    origin_twitte_id = set()  # set of origin_tweets_id
    tweet_user_dict = {}  # key: tweet id, value: user screen_name
    # user_tweet_dict = {}  # key: user screen_name, value: tweet id
    tweet_timestamp = {}  # key: tweet id, value: timestamp
    origin_retweet_dict = defaultdict(list)  # key: origin_tweet_id, vlaue: (created_at, retweet_id)
    quote_quoted_dict = {}

    cursor = relevant_tweet_col.find()
    # 初始化 tweet_user_dict  origin_twitte_id  origin_retweet_dict
    for document in cursor:
        # 过滤掉有错的推文
        if document.__contains__('errors'):
            continue

        id = document.get('id')
        if document.__contains__('user'):
            tweet_user_dict[id] = document.get('user').get('screen_name')
            # user_tweet_dict[document.get('user').get('screen_name')] = id

        created_at = document.get('created_at')  # 字符串时间，Fri Mar 13 20:18:44 +0000 2020
        if created_at == None:
            created_at_timestamp = None
        else:
            created_at_timestamp = time.mktime(time.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y"))  # 将字符串时间转换为时间戳

        tweet_timestamp[id] = created_at_timestamp

        if document.__contains__('retweeted_status'):
            tmp_id = document.get('retweeted_status').get('id')  # 原始推文id
            origin_twitte_id.add(tmp_id)
            origin_retweet_dict[tmp_id].append((created_at_timestamp, id))

        if document.__contains__('quoted_status'):
            quote = document['user']['screen_name']
            quoted = document['quoted_status']['user']['screen_name']
            quote_quoted_dict[quote] = quoted

    # print(origin_twitte_id)

    for key in origin_retweet_dict:
        origin_retweet_dict[key].sort()  # 将转推的推文列表按时间排序

    relevant_user_col = server.get_collection(user_col)

    # user_followers_dict = defaultdict(list)  # key:user_id, value: list of follower_id
    user_followers_dict = defaultdict(list)  # key:user_id, value: list of follower_screen_name
    user_followers_count_dict = defaultdict(int)  # key: user screen_name, value: followers_count
    user_cursor = relevant_user_col.find()
    # 初始化screen_namer_followers_dict, user_followers_count_dict
    for document in user_cursor:
        screen_name = document.get('screen_name')
        if document.__contains__('followers_screen_names'):
            user_followers_dict[screen_name] = document['followers_screen_names']
        if document.__contains__('followers_count'):
            user_followers_count_dict[screen_name] = document['followers_count']
        else:
            user_followers_count_dict[screen_name] = 0

    # 按followers_count由大到小对user_followers_count_dict排序
    followers_count_sorted_list = sorted(user_followers_count_dict.items(), key=lambda x: x[1], reverse=True)

    max_len = 0
    max_origin = ""
    for origin in origin_retweet_dict:
        if len(origin_retweet_dict[origin]) > max_len:
            max_len = len(origin_retweet_dict[origin])
            max_origin = origin

    # build net
    if max_origin != "":
        origin = max_origin
        G = nx.DiGraph()
        tmp_list = []
        index = 0
        if tweet_user_dict.__contains__(origin):
            if not G.has_node(tweet_user_dict[origin]):
                G.add_node(tweet_user_dict[origin], timestamp=str(tweet_timestamp[origin]),
                           tweet_id=str(origin), is_origin=True, is_quoted=False, is_quote=False, index=index)
                index += 1
            tmp_list.append(tweet_user_dict[origin])

        # 将所有转发者加入图中
        for timestamp, retweet_id in origin_retweet_dict[origin]:  # 遍历 原始推文
            retweet_user_screen_name = tweet_user_dict[retweet_id]  # 转发者的screen_name
            if not G.has_node(retweet_user_screen_name):  # 添加转发者节点
                G.add_node(retweet_user_screen_name,
                           timestamp=str(timestamp), tweet_id=str(retweet_id),
                           origin=str(origin), is_origin=False, is_quoted=False, is_quote=False, index=index)
                index += 1

            flag = False  # 记录转发者是否是前面某个用户的关注者
            if len(tmp_list) == 0:
                flag = True

            for user in tmp_list:
                # for j in range(len(tmp_list) - 1, -1, -1):
                #     user = tmp_list[j]
                if retweet_user_screen_name in user_followers_dict[user]:  # 如果转发者是前面某个用户的关注者
                    if G.has_edge(user, retweet_user_screen_name):
                        G.get_edge_data(user, retweet_user_screen_name)['weight'] += 1
                    else:
                        G.add_edge(user, retweet_user_screen_name, weight=1)
                    flag = True
                    break

            # 如果没有人的关注着列表里有它，就连在followers数最多的user上
            # if not flag:
            #     for user,count in followers_count_sorted_list:
            #         if user in tmp_list:
            #             G.add_edge(user, retweet_user_screen_name)
            #             break

            # 如果没有人的关注者列表里有它，就连根据followers数量按概率连接
            if not flag:
                count_list = [user_followers_count_dict[user] for user in tmp_list]
                norm_const = sum(count_list)
                if norm_const == 0:
                    normalized_probs = [1.0 / len(count_list) for count in count_list]
                else:
                    normalized_probs = [float(count) / norm_const for count in count_list]

                J, q = alias_setup(normalized_probs)
                user_index = alias_draw(J, q)
                user = tmp_list[user_index]
                if G.has_edge(user, retweet_user_screen_name):
                    G.get_edge_data(user, retweet_user_screen_name)['weight'] += 1
                else:
                    G.add_edge(user, retweet_user_screen_name, weight=1)

            tmp_list.append(retweet_user_screen_name)

        # for quote in quote_quoted_dict:
        #     quoted = quote_quoted_dict[quote]
        #     if quote == quoted:
        #         continue
        #     if G.has_node(quoted):
        #         if not G.has_node(quote):
        #             G.add_node(quote, timestamp=str(tweet_timestamp[user_tweet_dict[quote]]),
        #                        tweet_id=str(user_tweet_dict[quote]), is_origin=False, is_quote=True, is_quoted=False)
        #         G.nodes[quote]['is_quote'] = True
        #         G.nodes[quote]['is_quoted'] = True
        #         in_edges = list(G.in_edges(quote))
        #         G.remove_edges_from(in_edges)
        #         G.add_edge(quoted,quote)

        nx.write_gexf(G, "./tmp/{}_retweet_net.gexf".format(tweet_col))


def build_net_science(tweet_col, user_col, database):
    server = Mongodb("121.48.165.123", 30011, database, "readAnyDatabase", "Fzdwxxcl.121")
    relevant_tweet_col = server.get_collection(tweet_col)

    origin_twitte_id = set()  # set of origin_tweets_id
    tweet_user_dict = {}  # key: tweet id, value: user screen_name
    # user_tweet_dict = {}  # key: user screen_name, value: tweet id
    tweet_timestamp = {}  # key: tweet id, value: timestamp
    origin_retweet_dict = defaultdict(list)  # key: origin_tweet_id, vlaue: (created_at, retweet_id)
    quote_quoted_dict = {}

    cursor = relevant_tweet_col.find()
    # 初始化 tweet_user_dict  origin_twitte_id  origin_retweet_dict
    for document in cursor:
        # 过滤掉有错的推文
        if document.__contains__('errors'):
            continue

        id = document.get('id')
        if document.__contains__('user'):
            tweet_user_dict[id] = document.get('user').get('screen_name')
            # user_tweet_dict[document.get('user').get('screen_name')] = id

        created_at = document.get('created_at')  # 字符串时间，Fri Mar 13 20:18:44 +0000 2020
        if created_at == None:
            created_at_timestamp = None
        else:
            created_at_timestamp = time.mktime(time.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y"))  # 将字符串时间转换为时间戳

        tweet_timestamp[id] = created_at_timestamp

        if document.__contains__('retweeted_status'):
            tmp_id = document.get('retweeted_status').get('id')  # 原始推文id
            origin_twitte_id.add(tmp_id)
            origin_retweet_dict[tmp_id].append((created_at_timestamp, id))

        if document.__contains__('quoted_status'):
            quote = document['user']['screen_name']
            quoted = document['quoted_status']['user']['screen_name']
            quote_quoted_dict[quote] = quoted

    # print(origin_twitte_id)

    for key in origin_retweet_dict:
        origin_retweet_dict[key].sort()  # 将转推的推文列表按时间排序

    relevant_user_col = server.get_collection(user_col)

    # user_followers_dict = defaultdict(list)  # key:user_id, value: list of follower_id
    user_followers_dict = defaultdict(list)  # key:user_id, value: list of follower_screen_name
    user_followers_count_dict = defaultdict(int)  # key: user screen_name, value: followers_count
    user_cursor = relevant_user_col.find()
    # 初始化screen_namer_followers_dict, user_followers_count_dict
    for document in user_cursor:
        screen_name = document.get('screen_name')
        if document.__contains__('followers_screen_names'):
            user_followers_dict[screen_name] = document['followers_screen_names']
        if document.__contains__('followers_count'):
            user_followers_count_dict[screen_name] = document['followers_count']
        else:
            user_followers_count_dict[screen_name] = 0

    # 按followers_count由大到小对user_followers_count_dict排序
    followers_count_sorted_list = sorted(user_followers_count_dict.items(), key=lambda x: x[1], reverse=True)

    max_len = 0
    max_origin = ""
    for origin in origin_retweet_dict:
        if len(origin_retweet_dict[origin]) > max_len:
            max_len = len(origin_retweet_dict[origin])
            max_origin = origin

    # build net
    if max_origin != "":
        origin = max_origin
        G = nx.DiGraph()
        tmp_list = []
        index = 0
        if tweet_user_dict.__contains__(origin):
            if not G.has_node(tweet_user_dict[origin]):
                G.add_node(tweet_user_dict[origin], timestamp=str(tweet_timestamp[origin]),
                           tweet_id=str(origin), is_origin=True, is_quoted=False, is_quote=False, index=index)
                index += 1
            tmp_list.append(tweet_user_dict[origin])
        else:
            return

        # 将所有转发者加入图中
        for timestamp, retweet_id in origin_retweet_dict[origin]:  # 遍历 原始推文
            retweet_user_screen_name = tweet_user_dict[retweet_id]  # 转发者的screen_name
            if not G.has_node(retweet_user_screen_name):  # 添加转发者节点
                G.add_node(retweet_user_screen_name,
                           timestamp=str(timestamp), tweet_id=str(retweet_id),
                           origin=str(origin), is_origin=False, is_quoted=False, is_quote=False, index=index)
                index += 1

            flag = False  # 记录转发者是否是前面某个用户的关注者
            if len(tmp_list) == 0:
                flag = True

            for user in tmp_list:
                if retweet_user_screen_name in user_followers_dict[user]:  # 如果转发者是前面某个用户的关注者
                    if G.has_edge(user, retweet_user_screen_name):
                        G.get_edge_data(user, retweet_user_screen_name)['weight'] += 1
                    else:
                        G.add_edge(user, retweet_user_screen_name, weight=1)
                    flag = True
                    break

            # 如果没有人的关注着列表里有它，就连在followers数最多的user上
            # if not flag:
            #     for user,count in followers_count_sorted_list:
            #         if user in tmp_list:
            #             G.add_edge(user, retweet_user_screen_name)
            #             break

            # 如果没有人的关注者列表里有它，就连根据followers数量按概率连接
            if not flag:
                if G.has_edge(tweet_user_dict[origin], retweet_user_screen_name):
                    G.get_edge_data(tweet_user_dict[origin], retweet_user_screen_name)['weight'] += 1
                else:
                    G.add_edge(tweet_user_dict[origin], retweet_user_screen_name, weight=1)

            tmp_list.append(retweet_user_screen_name)

        # for quote in quote_quoted_dict:
        #     quoted = quote_quoted_dict[quote]
        #     if quote == quoted:
        #         continue
        #     if G.has_node(quoted):
        #         if not G.has_node(quote):
        #             G.add_node(quote, timestamp=str(tweet_timestamp[user_tweet_dict[quote]]),
        #                        tweet_id=str(user_tweet_dict[quote]), is_origin=False, is_quote=True, is_quoted=False)
        #         G.nodes[quote]['is_quote'] = True
        #         G.nodes[quote]['is_quoted'] = True
        #         in_edges = list(G.in_edges(quote))
        #         G.remove_edges_from(in_edges)
        #         G.add_edge(quoted,quote)

        nx.write_gexf(G, "./tmp/{}_retweet_net.gexf".format(tweet_col))


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    https://www.pythonheidong.com/blog/article/284697/
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()  # 移除并返回列表最后一个元素
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q


def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]


def build_followers_net_for_one_graph():
    with open('./pickle/user_followers', 'rb') as f:
        user_followers = pickle.load(f)
    with open('./pickle/user_id_screen_name', 'rb') as f:
        user_id_screen_name = pickle.load(f)
    with open('./pickle/screen_name_user_id', 'rb') as f:
        screen_name_user_id = pickle.load(f)

    graph = nx.read_gexf("./graph/tweet_1328493390472757253_retweet_net.gexf")
    nodes = graph.nodes()
    g = nx.DiGraph()
    count = 0
    node_count = 0
    for user in nodes:
        user_id = screen_name_user_id.get(user)
        if user_id is None:
            continue
        node_count += 1
        print(node_count)
        followers = user_followers[user_id]
        for follower in followers:
            follower_screen_name = user_id_screen_name.get(follower)
            if follower_screen_name is None:
                continue
            if not g.has_edge(user, follower_screen_name):
                count += 1
                print(f'add {count} edge')
                g.add_edge(user, follower_screen_name)  # 有意让边的方向为被关注者指向关注者，代表了信息流方向

    nx.write_gexf(g, './graph/tweet_253_followers_net.gexf')


def test():
    with open('./pickle/user_id_screen_name', 'rb') as f:
        user_id_screen_name = pickle.load(f)
    screen_name_user_id = {}
    for user in user_id_screen_name:
        screen_name_user_id[user_id_screen_name[user]] = user
    with open("./pickle/screen_name_user_id", "wb") as f:
        pickle.dump(screen_name_user_id, f)


def test1():
    g = nx.read_gexf("./graph/tweet_253_followers_net.gexf")
    print(len(g.nodes()))


if __name__ == '__main__':
    # build_net_by_id('tweet_1327425499472293888', 'multi_privacy', 'FE2020_AllTweets')
    # build_net_by_screen_name_split('KimberlyInfected', 'KimberlyInfectedUser', 'Privacy')
    # build_followers_net_for_one_graph()
    # test()
    test1()
