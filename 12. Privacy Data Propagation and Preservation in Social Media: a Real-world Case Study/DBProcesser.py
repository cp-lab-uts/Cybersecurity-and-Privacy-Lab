# -*- coding: utf-8 -*-
# @Time    : 2020/5/26 9:44
# @Author  : whm
# @Email   : whm0602@qq.com
# @File    : DBProcesser.py
# @Software: PyCharm

from Mongodb import Mongodb
# from dalib.target.twitter.TwitterApi import *
from dalib.lab.settings import *
from time import sleep
from concurrent.futures import *
import threading
from dalib.target.twitter import TwitterSelenium, TwitterApi
# from TwitterSelenium import TwitterSelenium
from dalib.lab.manager import TwitterApiManager
import argparse


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
        self.TAM = TwitterApiManager()
        self.api = TwitterApi(cm='twitterapi', proxy=ServerDict.proxy.new, **token)
        # self.api = TwitterApi(proxy=None, **token)

    def update_followers_ids(self, collection, start_from=0):
        """
        collection中必须要有Twitter user_id，该函数为user信息补全followers_ids信息
        :param start_from: 从第start_from + 1个document开始处理
        :param collection: 需要补全的collection
        :return:
        """
        collection = self.server.get_collection(collection)
        cursor = collection.find(no_cursor_timeout=True)
        count = 0
        try:
            for document in cursor:
                count += 1
                print(count)
                if count > start_from:
                    if not document.__contains__('followers_errors'):
                        if document.__contains__('followers_ids') and len(document['followers_ids']) > 0:
                            print('document has followers_ids already')
                        else:
                            followers_num = 0
                            if document.__contains__('followers_ids'):
                                followers_num = len(document['followers_ids'])
                            # id = document['id']
                            screen_name = document['screen_name']
                            followers_ids = []
                            cursor = -1
                            while True:
                                try:
                                    result = self.TAM.call('getFollowersIds', screen_name=screen_name, cursor=cursor)
                                except Exception as e:
                                    print(e)
                                    sleep(1)
                                    continue
                                # print(result)
                                if result.__contains__('ids'):
                                    followers_ids.extend(result.get('ids'))
                                    if len(followers_ids) > 500000:
                                        break
                                    cursor = result.get('next_cursor_str')

                                    if not cursor or int(cursor) == 0:
                                        break
                                else:
                                    if result.get('error') == 'Not authorized.':
                                        print(result['error'])
                                        collection.update_one({'screen_name': screen_name},
                                                              {'$set': {'followers_errors': 'Not authorized'}})

                                        break
                                    elif result.get('errors'):
                                        if result['errors'][0]['code'] == 34:
                                            print(result['errors'][0]['message'])
                                            collection.update_one({'screen_name': screen_name},
                                                                  {'$set': {'followers_errors': 'page not exists'}})

                                            break
                                        else:
                                            print(result)
                                            raise Exception('not known error')
                                    else:
                                        print(result)
                                        raise Exception('result has no ids')

                            if len(followers_ids) > followers_num:
                                followers_num = len(followers_ids)
                                collection.update_one({'screen_name': screen_name},
                                                      {'$set': {'followers_ids': followers_ids}})
                                print(f"user {screen_name} update {followers_num} followers")
                            else:
                                print('get less data!')
        finally:
            cursor.close()

    def get_followers_screen_names_from_tweet(self, tweet_col_name, followers_col_name):
        """
        从tweet_col_name中获取screen_name，然后调用__getfollowersname方法利用模拟浏览器获取对应的followers的screen_name，
        将信息插入到followers_ids_col_name中
        :param tweet_col_name: 推文collection
        :param followers_col_name: 保存followes信息的集合
        :return:
        """
        tweet_col = self.server.get_collection(tweet_col_name)
        followers_col = self.server.get_collection(followers_col_name)
        cursor = tweet_col.find()

        count = 0
        for document in cursor:
            # if count > 1:
            #     break
            if not document.__contains__('user'):  # 过滤推文中没有'user'字段的推文
                print("invalid document")
                continue

            screen_name = document['user']['screen_name']  # 从推文获取screen_name

            # 如果数据库中已经有了这个user，则不再重复获取
            if followers_col.find_one(
                    {'screen_name': screen_name, 'followers_screen_names': {'$exists': True}}) != None:
                print("{} already exist".format(screen_name))
                continue

            # 向线程池中提交任务
            self.thread_pool.submit(self.__getfollowersname, screen_name, followers_col)

            count += 1

    def get_followers_screen_names_from_user(self, followers_col_name, threshold=-1):
        """
        从followers_col_name中获取screen_name，然后调用__getfollowersname方法利用模拟浏览器获取对应的followers的screen_name，
        将信息插入到followers_col_name中
        :param followers_col_name:
        :return:
        """
        followers_col = self.server.get_collection(followers_col_name)
        cursor = followers_col.find()

        count = 0
        for document in cursor:
            if count <= threshold:
                count += 1
                continue
            # if count < 2000:
            #     count += 1
            #     continue
            # if count > 1:
            #     break
            if document.__contains__('errors'):
                print("invalid document")
                count += 1
                continue

            screen_name = document['screen_name']  # 从userobj获取screen_name

            # 如果数据库中已经有了这个user，则不再重复获取
            if document.__contains__('followers_screen_names'):
                print("{}. {}: followers_screen_names already exist".format(count, screen_name))
                count += 1
                continue

            # 向线程池中提交任务
            # self.thread_pool.submit(self.__getfollowersname, screen_name, followers_col, count)
            self.__getfollowersname(screen_name, followers_col, count)
            count += 1

    def __getfollowersname(self, screen_name, followers_col, index):
        """
        利用模拟浏览器获取screen_name用户的followers并插入到followers_col中
        :param screen_name:
        :param followers_col:
        :return:
        """
        print('index is {}'.format(index))
        print("thread {} starts".format(threading.currentThread().name))

        # 获取数据前判断数据是否已经存在，存在则跳过，不存在则插入
        if followers_col.find_one({'screen_name': screen_name, 'followers_screen_names': {'$exists': True}}) != None:
            print("{} already exists".format(screen_name))
            return

        obj = TwitterSelenium(proxy=ServerDict.proxy.new)  # 获取模拟浏览器类
        obj.login(**TokenDict.twitter.account)  # 账号扥附录
        print("start to get screen_name {}".format(screen_name))
        screen_names = obj.getFollowersNames(screen_name)  # 获取followers
        # print(screen_names)
        obj.quit()  # 获取完后关闭浏览器

        # 插入数据前判断数据是否已经存在，存在则跳过，不存在则插入
        if followers_col.find_one({'screen_name': screen_name, 'followers_screen_names': {'$exists': True}}) != None:
            print("{} already exists".format(screen_name))
        else:
            if followers_col.find_one({'screen_name': screen_name}) != None:
                followers_col.update_one({'screen_name': screen_name},
                                         {'$set': {'followers_screen_names': list(screen_names)}})
                print("{} update successfully".format(screen_name))
            else:
                followers_col.insert_one({'screen_name': screen_name,
                                          'followers_screen_names': list(screen_names)})
                print("{} insert successfully".format(screen_name))

    def update_user(self, user_col_name):
        """
        根据screen_name更新user表的内容，加上user更详细的信息，要求原表中存在screen_name字段
        :param user_col_name: 要更新的collection名
        :return:
        """
        col = self.server.get_collection(user_col_name)
        result = col.find()
        screen_name_list = []
        count = 0
        for document in result:
            count += 1
            print(count)
            if document.__contains__('id'):
                print('{} 信息已完整'.format(document['id']))
                continue

            if document.__contains__('screen_name'):
                if document.__contains__('followers_errors'):
                    print(f'{document["screen_name"]} is {document["followers_errors"]}')
                    continue
                screen_name_list.append(document['screen_name'])
            else:
                continue
            if len(screen_name_list) == 100:
                user_objs = self.TAM.call('getUsers', screen_name=screen_name_list)
                # print(user_objs)
                for user_obj in user_objs:
                    if user_obj.__contains__('errors'):
                        print(user_obj)
                        continue
                    if col.find_one({'id': user_obj['id']}) != None:
                        print('{} 信息已完整'.format(user_obj['id']))
                        continue
                    col.update_one({'screen_name': user_obj['screen_name']}, {'$set': user_obj})
                print('update 100 users successfully')
                screen_name_list = []

        # 最后一次循环没有达到100个，没有更新完全，需补上
        if len(screen_name_list) != 0:
            user_objs = self.TAM.call('getUsers', screen_name=screen_name_list)
            # print(user_objs)
            for user_obj in user_objs:
                if user_obj.__contains__('errors'):
                    print(user_obj)
                    continue
                if col.find_one({'id': user_obj['id']}) != None:
                    print('{} 信息已完整'.format(user_obj['id']))
                    continue
                col.update_one({'screen_name': user_obj['screen_name']}, {'$set': user_obj})
        print('update all users successfully')

    def insert_user(self, from_col_name, to_col_name):
        """
        从from_col中的推文中获取推文相关的user，包括推文的作者，
        推文引用的原推文作者，推文转发的推文原作者，存入to_col
        中
        :param from_col_name: 推文collection
        :param to_col_name:
        :return:
        """
        from_col = self.server.get_collection(from_col_name)
        to_col = self.server.get_collection(to_col_name)

        documents = from_col.find()
        screen_name_list = []
        for document in documents:
            if document.__contains__('errors'):
                continue
            screen_name = document['user']['screen_name']
            if to_col.find_one({'screen_name': screen_name}) == None:
                screen_name_list.append(screen_name)

            if document.__contains__('quoted_status'):
                quoted_user = document['quoted_status']['user']['screen_name']
                if to_col.find_one({'screen_name': quoted_user}) == None:
                    screen_name_list.append(quoted_user)

            if document.__contains__('retweeted_status'):
                retweeted_user = document['retweeted_status']['user']['screen_name']
                if to_col.find_one({'screen_name': retweeted_user}) == None:
                    screen_name_list.append(retweeted_user)

            if len(screen_name_list) >= 90:
                user_objs = self.api.getUsers(screen_name=screen_name_list)
                # print(user_objs)
                for user_obj in user_objs:
                    if user_obj.__contains__('errors'):
                        print(user_obj['errors'])
                    if to_col.find_one({'screen_name': user_obj['screen_name']}) != None:
                        print('{} already exists'.format(user_obj['screen_name']))
                        continue
                    to_col.insert_one(user_obj)
                    print('insert successfully')
                screen_name_list = []

        if len(screen_name_list) != 0:
            user_objs = self.api.getUsers(screen_name=screen_name_list)
            # print(user_objs)
            for user_obj in user_objs:
                if user_obj.__contains__('errors'):
                    print(user_obj['errors'])
                if to_col.find_one({'screen_name': user_obj['screen_name']}) != None:
                    print('{} already exists'.format(user_obj['screen_name']))
                    continue

                to_col.insert_one(user_obj)
                print('insert successfully')

    def move(self, from_col_name, to_col_name):
        from_col = self.server.get_collection(from_col_name)
        to_col = self.server.get_collection(to_col_name)
        documents = from_col.find()
        count = 0
        for document in documents:
            to_col.insert_one(document)
            count += 1

    def insert_origin_tweet(self, to_col_name):
        """
        补充to_col中的推文的retweeted_status中的推文和quoted_status中的推文
        :param to_col_name:
        :return:
        """
        to_col = self.server.get_collection(to_col_name)
        documents = to_col.find()
        tweet_ids_list = []
        for document in documents:
            if document.__contains__('errors'):
                continue

            if document.__contains__('quoted_status'):
                id = document['quoted_status']['id']
                if to_col.find_one({'id': id}) is None:
                    tweet_ids_list.append(id)

            if document.__contains__('retweeted_status'):
                id = document['retweeted_status']['id']
                if to_col.find_one({'id': id}) is None:
                    tweet_ids_list.append(id)

            if len(tweet_ids_list) >= 90:
                tweet_objs = self.api.getTweetsByIds(ids=tweet_ids_list)
                # print(user_objs)
                for tweet_obj in tweet_objs:
                    if tweet_obj.__contains__('errors'):
                        print(tweet_obj['errors'])
                    if to_col.find_one({'id': tweet_obj['id']}) != None:
                        print('{} already exists'.format(tweet_obj['id']))
                        continue
                    to_col.insert_one(tweet_obj)
                    print('insert successfully')
                tweet_ids_list = []

        if len(tweet_ids_list) != 0:
            tweet_objs = self.api.getTweetsByIds(ids=tweet_ids_list)
            # print(user_objs)
            for tweet_obj in tweet_objs:
                if tweet_obj.__contains__('errors'):
                    print(tweet_obj['errors'])
                if to_col.find_one({'id': tweet_obj['id']}) != None:
                    print('{} already exists'.format(tweet_obj['id']))
                    continue
                to_col.insert_one(tweet_obj)
                print('insert successfully')

    def copy_retweet_to(self, tweet_id, from_col_name, to_col_name):
        from_col = self.server.get_collection(from_col_name)
        to_col = self.server.get_collection(to_col_name)

        documents = from_col.find({'id': tweet_id})
        for document in documents:
            if to_col.find_one({'id': tweet_id}) is None:
                to_col.insert_one(document)

        documents = from_col.find({'retweeted_status.id': tweet_id})
        for document in documents:
            if to_col.find_one({'id': document['id']}) is None:
                to_col.insert_one(document)
                print("insert successfully")

    def copy_documents_to(self, from_col_name, to_col_name):
        """
        将from_col中的documents拷贝到to_col_name
        :param from_col_name:
        :param to_col_name:
        :return:
        """
        from_col = self.server.get_collection(from_col_name)
        to_col = self.server.get_collection(to_col_name)
        documents = from_col.find()
        for document in documents:
            if to_col.find_one(document) == None:
                to_col.insert_one(document)
        print("copy finished")

    # def __del__(self):
    #     """
    #     释放资源
    #     :return:
    #     """
    #     self.thread_pool.shutdown()

    def get_followers_information(self, user_col_name):
        col = self.server.get_collection(user_col_name)
        result = col.find()
        screen_name_list = []
        for document in result:
            if document.__contains__('id'):
                print('{} 信息已完整'.format(document['id']))
                continue
            if document.__contains__('screen_name'):
                screen_name_list.append(document['screen_name'])
            else:
                continue
            if len(screen_name_list) == 100:
                user_objs = self.api.getUsers(screen_name=screen_name_list)
                # print(user_objs)
                for user_obj in user_objs:
                    if user_obj.__contains__('errors'):
                        print(user_obj['errors'])
                    if col.find_one({'id': user_obj['id']}) != None:
                        print('{} 信息已完整'.format(user_obj['id']))
                        continue
                    col.update_one({'screen_name': user_obj['screen_name']}, {'$set': user_obj})
                screen_name_list = []

        # 最后一次循环没有达到100个，没有更新完全，需补上
        if len(screen_name_list) != 0:
            user_objs = self.api.getUsers(screen_name=screen_name_list)
            # print(user_objs)
            for user_obj in user_objs:
                if user_obj.__contains__('errors'):
                    print(user_obj['errors'])
                if col.find_one({'id': user_obj['id']}) != None:
                    print('{} 信息已完整'.format(user_obj['id']))
                    continue
                col.update_one({'screen_name': user_obj['screen_name']}, {'$set': user_obj})


def run_update_followers_ids(database, col, start_from):
    dbp = DBProcesser(username='dlx', password='kb314dlx', database=database)
    dbp.update_followers_ids(col, start_from=start_from)


def run_update_user(database, col):
    dbp = DBProcesser(username='dlx', password='kb314dlx', database=database)
    dbp.update_user(col)


def parse_args():
    '''
    Parses the DBProcesser arguments.
    '''
    parser = argparse.ArgumentParser(description="Run DBprocesser.")

    parser.add_argument('-f', '--update-followers-ids', dest='opt', action='store_const',
                        const=run_update_followers_ids,
                        help='run update-followers-ids')
    parser.add_argument('-u', '--update-user', dest='opt', action='store_const',
                        const=run_update_user,
                        help='run update-user')
    parser.add_argument('--db', default='FE2020_AllTweets', help='dbname, 默认值为FE2020_AllTweets')
    parser.add_argument('--col', default='test', help='col, 默认值为test')
    parser.add_argument('--count', type=int, default=0, help='count for update-followers-ids')

    return parser


def main():
    parser = parse_args()
    args = parser.parse_args()
    if args.opt == run_update_user:
        args.opt(args.db, args.col)
    elif args.opt == run_update_followers_ids:
        args.opt(args.db, args.col, args.count)
    else:
        parser.print_help()
        return


if __name__ == '__main__':
    main()
    # print(args)
    # dbp = DBProcesser(username='dlx', password='kb314dlx', database=args.db)
    # if args.update_followers_ids:
    #     dbp.update_followers_ids('test')
    # elif args.update_user:
    #     dbp.update_user('test')
    # dbp = DBProcesser(username='dlx', password='kb314dlx', database='FE2020_AllTweets')
    # dbp.update_followers("RelevantUser")
    # dbp.update_followers_ids("RumorUser")
    # dbp.get_followers_screen_names_from_tweet("Rumor", "RumorUser")
    # dbp.move('riot_200609', 'Riot')
    # dbp.insert_user('WoodwardExposesKavanaugh', 'WoodwardExposesKavanaughUser')
    # dbp.insert_user('TrumpStroken', 'TrumpStrokenUser')  # 采推文的用户
    # dbp.get_followers_screen_names_from_user('TomParkerbraintumourUser')  # 添加followers_screen_names字段
    # dbp.get_followers_screen_names_from_user('TwitterControlledUser')
    # dbp.insert_origin_tweet('Rumor')
    # dbp.copy_retweet_to(1270023344491020291, 'riot_200610', 'Riot2')
    # dbp.copy_documents_to('TwistedLydaKrewson', 'TwistedLydaKrewson_before_200629')
    # dbp.update_followers_ids('test', start_from=50000)
    # dbp.update_user('test')
