from ..toolkit.utils.data_file import DataFile
from ..nlp_tools.text.text_utils import sentence_cut
from ..nlp_tools.word.word_utils import cut
from ..coin_recgnition.coin_recognition import CoinRecognition
import os
import math
from collections import defaultdict

class MarketConditionAnalysis:

    def __init__(self):
        words_dict = dict()
        for lang in ('en','zh','ru'):
            words_dict[lang] = dict()
            words_dict[lang]['pos'],words_dict[lang]['neg'],words_dict[lang]['pri'] = self.load_word_dicts(lang)
        self.word_dict = words_dict
        self.cr = CoinRecognition()

    def load_word_dicts(self,lang):
        dir_ = os.path.dirname(__file__)
        pos_words_path = os.path.join(dir_,'word_dict',f'pos_words_{lang}.txt')
        neg_words_path = os.path.join(dir_, 'word_dict', f'neg_words_{lang}.txt')
        pri_words_path = os.path.join(dir_, 'word_dict', f'pri_words_{lang}.txt')

        pos_words = DataFile.load_words_set(pos_words_path)
        neg_words = DataFile.load_words_set(neg_words_path)
        pri_words = DataFile.load_words_set(pri_words_path)
        return pos_words,neg_words,pri_words

    def get_coin_conditions(self,text,lang):
        '''
        对文本进行行情分析，返回行情得分（涨1.0  --  -1.0跌）
        :param text:
        :param lang:
        :return: 行情得分 -1.0   ---  1.0
        '''
        assert lang in ('en','zh','ru') ,Exception('only support en,zh ,ru language')
        value_word_count = 0
        word_count = 0
        #total_score = 0
        coin_score = defaultdict(float)
        coin_value_word_count = defaultdict(int)
        for sentence in sentence_cut(text,'complete'):
            last_subsentence_score = 0
            last_coins = set()
            last_subsentence_value_word_count = 0
            for subsentence in sentence_cut(sentence,sep = 'uncomplete'):
                sub_coins ,_ = self.cr.find_coins_in_text(subsentence,lang)
                sub_coins = set(sub_coins) if sub_coins else set()
                sub_sentence_value_word_count = 0

                sub_sentence_score = 0.0
                pri_count = 0
                r = 0.1
                for word in cut(subsentence,lang=lang):
                    word_count+=1
                    if word in self.word_dict[lang]['pos']:
                        sub_sentence_score+=1
                        sub_sentence_value_word_count+=1
                    elif word in self.word_dict[lang]['neg']:
                        sub_sentence_score-=1
                        sub_sentence_value_word_count+=1
                    elif word in self.word_dict[lang]['pri']:
                        pri_count+=1
                for i in range(pri_count):
                    sub_sentence_score*=-1


                coins = sub_coins&last_coins if not sub_coins else sub_coins
                sentence_value_word_count = sub_sentence_value_word_count+last_subsentence_value_word_count if not sub_sentence_value_word_count else sub_sentence_value_word_count
                sentence_score = sub_sentence_score+last_subsentence_score if not sub_sentence_score else sub_sentence_score

                if coins and sentence_value_word_count: #可以与上次的组合，合并为一句处理
                    for coin in coins:
                        coin_score[coin]+=sentence_score
                        coin_value_word_count[coin]+=sentence_value_word_count
                        value_word_count+=sentence_value_word_count
                    last_subsentence_score = 0
                    last_coins = set()
                    last_subsentence_value_word_count = 0
                else: #无法组合，处理上半句，并将本半句更新为上半句
                    last_coins = ['unknow'] if not last_coins and value_word_count else last_coins
                    for coin in last_coins:
                        coin_score[coin]+=last_subsentence_score
                        coin_value_word_count[coin]+=last_subsentence_value_word_count
                        value_word_count+=last_subsentence_value_word_count
                    last_subsentence_score = 0
                    last_coins = set()
                    last_subsentence_value_word_count = 0

                #循环结束处理最后半句
                if last_coins or last_subsentence_value_word_count:
                    last_coins = ['unknow'] if not last_coins and value_word_count else last_coins
                    for coin in last_coins:
                        coin_score[coin] += last_subsentence_score
                        coin_value_word_count[coin] += last_subsentence_value_word_count
                        value_word_count += last_subsentence_value_word_count

        ret = defaultdict(float)
        for coin ,score in coin_score.items():
            value_word_count_coin = coin_value_word_count[coin]
            if score:
                ret[coin] = score / value_word_count_coin * math.pow(value_word_count_coin, r) / math.pow(word_count, r)
            else:
                ret[coin] = 0.0
        return ret
        #print(total_score,value_word_count,word_count,math.pow(value_word_count,0.02)/math.pow(word_count,0.02))

        #return total_score/value_word_count*math.pow(value_word_count,r)/math.pow(word_count,r)



