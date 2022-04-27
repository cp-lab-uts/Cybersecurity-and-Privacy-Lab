from Mongodb import Mongodb
from dalib.lab.shortcut.mongo import FE2020
import time


def create_index_in_db():
    server = Mongodb("121.48.165.123",
                     30011,
                     "FE2020",
                     "dlx",
                     "kb314dlx")

    obj = FE2020()
    all_tbs = obj.list()
    tbs = list(filter(lambda name: name.startswith('Final'), all_tbs))
    tbs.remove('Final')
    tbs = sorted(tbs, key=lambda name: int(name.split('_')[-1]), reverse=True)

    for col_name in tbs:
        print("正在操作" + col_name + "中的数据")
        col = server.get_collection(col_name)
        col.ensure_index("retweeted_status.id")


def create_index_file():
    server = Mongodb("121.48.165.123",
                     30011,
                     "Privacy",
                     "dlx",
                     "kb314dlx")

    # obj = FE2020()
    # all_tbs = obj.list()
    # tbs = list(filter(lambda name: name.startswith('Final'), all_tbs))
    # tbs.remove('Final')
    # tbs = sorted(tbs, key=lambda name: int(name.split('_')[-1]), reverse=False)
    # for name in tbs.copy():
    #     if int(name.split('_')[-1]) > 201126 or int(name.split('_')[-1]) < 201109:
    #         tbs.remove(name)
    tbs = ['KimberlyInfected', 'MaryTrump', 'TrumpSAT']
    for col_name in tbs:
        print("正在操作" + col_name + "中的数据")
        index_file = open(f"./index/{col_name}_index.txt", "w")
        col = server.get_collection(col_name)
        count = 0
        for doc in col.find():
            count += 1
            print(count)
            id = doc['id']
            retweeted_id = 0
            if doc.__contains__("retweeted_status"):
                retweeted_id = doc['retweeted_status']['id']
            screen_name = doc['user']['screen_name']
            created_at_timestamp = 0
            created_at = doc.get('created_at')  # 字符串时间，Fri Mar 13 20:18:44 +0000 2020
            if created_at != None:
                created_at_timestamp = time.mktime(
                    time.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y"))  # 将字符串时间转换为时间戳

            followers_count = doc['user']['followers_count']

            index_file.write(str(id) + '\t' + str(retweeted_id) + '\t' +
                             screen_name + '\t' + str(created_at_timestamp) +
                             '\t' + str(followers_count) + '\n')

        index_file.close()


def create_screenname_followers_count_index_from_user_col():
    server = Mongodb("121.48.165.123",
                     30011,
                     "Privacy",
                     "dlx",
                     "kb314dlx")

    tbs = ['KimberlyInfectedUser', 'MaryTrumpUser', 'TrumpSATUser']
    for col_name in tbs:
        print("正在操作" + col_name + "中的数据")
        index_file = open(f"./index/{col_name}_index.txt", "w")
        col = server.get_collection(col_name)
        count = 0
        for doc in col.find():
            count += 1
            print(count)
            screen_name = doc['screen_name']
            followers_count = doc['followers_count']

            index_file.write(screen_name + '\t' + str(followers_count) + '\n')

        index_file.close()


if __name__ == "__main__":
    # create_index_file()
    create_screenname_followers_count_index_from_user_col()
