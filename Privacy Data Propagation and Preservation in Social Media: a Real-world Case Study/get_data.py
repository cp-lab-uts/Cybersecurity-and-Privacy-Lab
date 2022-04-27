from dalib.target.twitter import TwitterApi
from dalib.lab.settings import *
from concurrent.futures import *
import networkx as nx
from Translator import Translator
import pickle
from tqdm import tqdm


def test1():
    translator = Translator()

    token = TokenDict.twitter.default
    api = TwitterApi(cm='twitterapi', proxy=ServerDict.proxy.new, **token)

    g = nx.read_gexf('./graph/Group1.gexf')
    nodes = list(g.nodes())

    docs = []
    for i in range(0, len(nodes), 100):
        if i + 100 <= len(nodes):
            docs.extend(api.getTweets(nodes[i:i + 100]))
    f = open('txt/origin_tweets_1.txt', 'w', encoding='utf8')
    for doc in docs:
        id = doc['id']
        full_text = doc['full_text']
        tmplist = full_text.split()
        full_text = ' '.join(tmplist)
        tmpstr = translator.translate(full_text)
        f.write(f'{id}\t{full_text}\t{tmpstr}\n')
    f.close()

# test2-5 is for project tweet classification
def test2():
    translator = Translator()

    token = TokenDict.twitter.default
    api = TwitterApi(cm='twitterapi', proxy=ServerDict.proxy.new, **token)

    with open("./pickle/tweets_collect", "rb") as f:
        tweets_collect = pickle.load(f)

    tweets_collect = list(tweets_collect)
    docs = []
    for i in range(0, len(tweets_collect), 100):
        if i + 100 <= len(tweets_collect):
            docs.extend(api.getTweets(tweets_collect[i:i + 100]))
            i += 100
        else:
            docs.extend(api.getTweets(tweets_collect[i:]))
            break
    f = open('txt/tweet_collect.txt', 'w', encoding='utf8')
    for doc in docs:
        id = doc['id']
        full_text = doc['full_text']
        tmplist = full_text.split()
        full_text = ' '.join(tmplist)
        # tmpstr = translator.translate(full_text)
        # f.write(f'{id}\t{full_text}\t{tmpstr}\n')
        f.write(f'{id}\t{full_text}\n')
    f.close()

def test3():
    translator = Translator()
    f = open('txt/tweet_collect.txt', 'r', encoding='utf8')
    lines = list(f.readlines())
    f.close()
    f = open('txt/tweets_translate.txt', 'w', encoding='utf8')
    pbar = tqdm(total=len(lines))
    for doc in lines:
        doc = doc.strip()
        doc = doc.split('\t')
        full_text = doc[1][:-1]
        tmplist = full_text.split('\t')
        full_text = ' '.join(tmplist)
        tmpstr = translator.translate(full_text)
        f.write(f'{doc[0]}\t{full_text}\t{tmpstr}\n')
        pbar.update(1)
    pbar.close()
    f.close()

def test4():
    f = open('txt/tweet_collect.txt', 'r', encoding='utf8')
    lines = list(f.readlines())
    f.close()
    f = open('txt/tweets_translate.txt', 'r', encoding='utf8')
    trans = list(f.readlines())
    f.close()
    f = open('txt/tweets_translate_new.txt', 'w', encoding='utf8')
    pbar = tqdm(total=len(lines))
    for i, doc in enumerate(lines):
        doc = doc.strip()

        full_text = trans[i].strip()
        tmplist = full_text.split('\t')
        full_text = tmplist[-1]
        f.write(f'{doc}\t{full_text}\n')
        pbar.update(1)
    pbar.close()
    f.close()

def test5():
    with open("./pickle/Trump", "rb") as f:
        trump = pickle.load(f)
    with open("./pickle/Biden", "rb") as f:
        biden = pickle.load(f)

    f = open('txt/tweets_translate_new.txt', 'r', encoding='utf8')
    f1 = open('Trump.txt', 'w', encoding='utf8')
    f2 = open('Biden.txt', 'w', encoding='utf8')
    for line in f.readlines():
        id = line.split()[0]
        id = int(id)
        if id in trump:
            f1.write(line)
        elif id in biden:
            f2.write(line)
        else:
            raise Exception("error")
    f2.close()
    f1.close()
    f.close()



if __name__ == "__main__":
    test5()
