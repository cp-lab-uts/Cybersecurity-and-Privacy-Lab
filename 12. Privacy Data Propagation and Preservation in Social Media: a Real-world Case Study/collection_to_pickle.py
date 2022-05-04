import pickle
from Mongodb import Mongodb
import networkx as nx


def find_removed_tweets():
    f = open('txt/origin_tweets_1.txt', 'r', encoding='utf8')
    tweet_ids = []
    for line in f.readlines():
        a = line.split()
        tweet_ids.append(int(a[0]))
    f.close()

    removed_tweets_id = []
    g = nx.read_gexf("graph/Group1.gexf")
    nodes = g.nodes()
    for node in nodes:
        if node not in tweet_ids:
            removed_tweets_id.append(int(node))

    # files = os.listdir('index_data')
    # tweet_id_col_name_dict = {}
    # for file in files:
    #     path = os.path.join('index_data', file)
    #     with open(path, 'r') as f:
    #         for line in f.readlines():
    #             a = line.split()
    #             tweet_id_col_name_dict[int(a[0])] = file[:-10]
    # with open('./pickle/all_tweet_id_col_name', 'rb') as f:
    #     tweet_id_col_name_dict = pickle.load(f)
    #
    # removed_id_col_name_dict = {}
    # count = 0
    # for removed_id in removed_tweets_id:
    #     count += 1
    #     print(count)
    #     if tweet_id_col_name_dict.get(removed_id):
    #         removed_id_col_name_dict[removed_id] = tweet_id_col_name_dict[removed_id]
    #
    # with open('./pickle/removed_id_col_name', 'wb') as f:
    #     pickle.dump(removed_id_col_name_dict, f)
    with open('./pickle/removed_id_col_name', 'rb') as f:
        removed_id_col_name_dict = pickle.load(f)

    f = open('./removed_tweets.txt', 'w', encoding='utf8')
    server = Mongodb("121.48.165.123", 30011, 'FE2020', "readAnyDatabase", "Fzdwxxcl.121")
    for removed_id in removed_id_col_name_dict:
        colname = removed_id_col_name_dict[removed_id]
        col = server.get_collection(colname)
        doc = col.find_one({'id': removed_id})
        text = doc.get("full_text") or doc.get("text")
        text = text.replace('\n', ' ')
        text = text.replace('\r\n', ' ')
        text = text.replace('\t', ' ')
        f.write(f'{removed_id}\t{text}\n')
    f.close()


if __name__ == '__main__':
    # pickle_user_followers('multi_privacy')
    cols = ['tweet_1327425499472293888_User', 'tweet_1327940661258162176_User',
            'tweet_1327971951852343297_User', 'tweet_1328402814800977920_User',
            'tweet_1328452034450907138_User', 'tweet_1328493390472757253_User',
            'tweet_1329107891920527362_User']
    # find_union_users(cols)
    find_removed_tweets()
