from ..toolkit.utils.data_file import DataFile
from ..nlp_tools.word.word_utils import pos_tag
from collections import defaultdict
import os
def __tag_rule_zh__(cls,tag):
    s = set(['eng', 'n'])
    return tag in s


def __tag_rule_en__(cls, tag):
    s = set(['NN','NNS','NNP','NNPS'])
    return tag in s

def __tag_rule_ru__(cls, tag):
    s = set(['S','NONLEX'])
    return tag in s

__tag_rule__ = dict()
__tag_rule__['en'] = __tag_rule_en__
__tag_rule__['zh'] = __tag_rule_zh__
__tag_rule__['ru'] = __tag_rule_ru__
class CoinRecognition:
    not_lower = ['PEOPLE','NEW',
                 'ONE','FLOW','VALUE','TALK','VISION',
                 'MEET','MAN','NODE','SUN','BIT','DOGE','CAKE',
                 'NEST','BOX','MASK','WAR']
    config_dir = os.path.dirname(__file__)
    alternative_dict = DataFile.load_str_dict(os.path.join(config_dir,'alternative_coin_names.txt'),split = ',',lower_k=True,lower_v=True)
    coins_set = DataFile.load_words_set(os.path.join(config_dir,'coins.txt'))
    _coins_set = set()
    coin_dict = dict()
    for coin in coins_set:
        coin_dict[coin] = coin
        _coins_set.add(coin+'s')
        coin_dict[coin+'s'] = coin
        if coin not in not_lower:
            _coins_set.add(coin.lower())
            coin_dict[coin.lower()] = coin
            _coins_set.add(coin.lower()+'s')
            coin_dict[coin.lower()+'s'] = coin

    coins_set = coins_set|_coins_set
    del _coins_set



    def __init__(self):

        pass




    def find_coins_in_text(self,text,lang):
        '''
        返回文本包含的最主要币种，和每个币种的出现频次
        :param text: 文本
        :param lang: 语言
        :return: 最主要币种 , 币种频次={'币种1':(出现频次，频次占比),'币种2':(出现频次，频次占比),...}
        '''
        assert lang in ('en','zh','ru') ,Exception("only ('en','zh','ru') are supported")
        l = pos_tag(text, lang=lang)
        l = filter(lambda x:__tag_rule__[lang](self,x[1]),l)
        words = [w[0] if w[0] not in self.alternative_dict else self.alternative_dict[w[0]] for w in l]
        ret =  defaultdict(int)

        for w in words:
            if w in self.coins_set:
                w = self.coin_dict[w]
                ret[w]+=1
        if ret:
            coins = list(ret.keys())
            freqs = list(ret.values())
            total = sum(freqs)
            max_freq = max(freqs)
            #max_coins = coins[freqs.index(max_freq)]
            max_coins = [coins[i] for i ,j in enumerate(freqs) if j == max_freq]
            ret = {w: (f, float(f) / total) for w, f in ret.items()}
            return max_coins, ret
        else:
            return None, None








