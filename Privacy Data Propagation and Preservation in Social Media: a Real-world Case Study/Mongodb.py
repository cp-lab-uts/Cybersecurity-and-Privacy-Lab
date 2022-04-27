from pymongo import MongoClient

class Mongodb:
    def __init__(self, ip, port, db_name, name, password):
        self.connection = MongoClient(ip, port)
        self.connection.get_database('admin').authenticate(name, password)
        self.database = self.connection.get_database(db_name)

        # try:
        #     self.database = self.connection[dbname]
        # except BaseException:
        #     print("no such database")

    def get_collection(self, col_name):
        return self.database.get_collection(col_name)
def get_description():
    server = Mongodb("121.48.165.123", 30011, "Group_Account", "readAnyDatabase", "Fzdwxxcl.121")
    col_names = ["think_tank_20190121", "political_20190121","organization_20190121", "spam_account",
                 "defence_20190121", "news_reporter_20190121"]
    fd = open("twitter_accounts_description_label2.txt", "w", encoding='utf-8')
    result_count = []
    for index, col_name in enumerate(col_names):
        col = server.get_collection(col_name)
        count = 0
        for document in col.find():
            print(count)
            if document.__contains__("description"):
                tmpstr = document["description"]
                tmpstr = tmpstr.strip()
                tmpstr = tmpstr.replace('\r\n',' ')
                tmpstr = tmpstr.replace('\r',' ')
                tmpstr = tmpstr.replace('\n',' ')
                if(tmpstr == ""):
                    print("empty string" + str(count))
                    continue

                # print(document["description"])
                fd.write(tmpstr + "\t" + str(index) + "\n")
                count+=1
            else:
                print("docunment has no decription")
        print(col_name + ":" + str(count))
        result_count.append(count)

    fd.close()
    print(result_count)

if __name__ == "__main__":
    # server = Mongodb("121.48.165.123", 30011, "Group_Account", "readAnyDatabase", "Fzdwxxcl.121")
    # col = server.get_collection("test_4_26")
    s2 = Mongodb("121.48.165.123", 30011, "Privacy", "dlx", "kb314dlx")
    # col2 = s2.get_collection('test')
    col2 = s2.get_collection('KimberlyInfected')
    # a = col2.find_one({'id':1279238243410604033})
    count = 0
    for document in col2.find():
        if document['id'] == 1279238243410604033:
            continue
        if document.__contains__('retweeted_status'):
            if(document['retweeted_status']['id'] == 1279238243410604033):
                count += 1
                print(count)
            else:
                col2.delete_one(document)
        else:
            col2.delete_one(document)

    # 必须使用with语句，否则出现异常后cursor不关闭在服务器中占用资源，当然也可使用try except finally
    # with col.find({'text':{'$regex': 'Tasuku Honjo'}}, no_cursor_timeout=True).batch_size(200) as result:
    #     count = 0
    #     for document in result:
    #         count+= 1
    #         print(count)
    #         id = document['id']
    #         if col2.find_one({'id': id}) != None:
    #             print("{} alreay exists".format(id))
    #             continue
    #
    #         col2.insert_one(document)
    #         print("{} insert successfully".format((id)))
    # '''
    #     for document in col.find():
    #     document['']
    # '''
    # for document in col.find():
    #     document['']