import requests

class Translator:
    def __init__(self):
        self.url = "http://fanyi.youdao.com/translate"
        # self.url = "https://translate.google.cn"
    def translate(self, string):
        data = {
            'doctype': 'json',
            'type': 'AUTO',
            'i': string
        }
        # data = {
        #     'sl': 'auto',
        #     'tl': 'zh-CN',
        #     'text': string,
        #     'op' : 'translate'
        # }
        r = requests.get(self.url, params=data)
        # r = requests.get(self.url)
        result = r.json()
        target = []
        for tmp in result['translateResult'][0]:
            target.append(tmp['tgt'])
        return " ".join(target)

if __name__ == '__main__':
    t = Translator()
    r = t.translate("A large group of demonstrators gathered outside the Staten Island bar that was shut down by sheriff's deputies for violating COVID restrictions")
    print(r)