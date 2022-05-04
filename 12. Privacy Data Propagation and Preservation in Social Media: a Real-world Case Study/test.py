import pymysql
from Mongodb import Mongodb
from dalib.lab.shortcut.mongo import FE2020
import time

conn = pymysql.connect('192.168.31.169', user='root', passwd="ming", db="tweets")
print(conn)
cursor = conn.cursor()
server = Mongodb( "121.48.165.123",
                              30011,
                              "FE2020",
                              "readAnyDatabase",
                              "Fzdwxxcl.121")

obj = FE2020()
all_tbs = obj.list()
tbs = list(filter(lambda name: name.startswith('Final'), all_tbs))
tbs.remove('Final')
tbs = sorted(tbs, key=lambda name: int(name.split('_')[-1]), reverse=True)

for col_name in tbs:
    print("正在查询" + col_name + "中的数据")
    col = server.get_collection(col_name)
    for doc in col.find():
        id  = doc['id']
        screen_name = doc['user']['screen_name']
        retweet_id = 0
        if doc.__contains__('retweeted_status'):
            retweet_id = doc['retweeted_status']['id']

        created_at_timestamp = 0
        created_at = doc.get('created_at')  # 字符串时间，Fri Mar 13 20:18:44 +0000 2020
        if created_at != None:
            created_at_timestamp = time.mktime(time.strptime(created_at, "%a %b %d %H:%M:%S +0000 %Y"))  # 将字符串时间转换为时间戳

        query = f"insert into tweets values({id}, \"{screen_name}\",{retweet_id}, {created_at_timestamp});"
        cursor.execute(query)
        conn.commit()

cursor.close()

conn.close()
