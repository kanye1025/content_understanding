# -*- coding:utf-8 -*-
"""
资产币种分类
"""
import logging
import os
import re
import unicodedata
import fasttext
import jieba
import jieba.analyse
import jieba.posseg
import nltk
import openpyxl
#import pyspark.sql.functions as Func
#from pyspark.sql.types import *

from pallas.frameworks.content_understanding.classify.config import DictFilePath, LanguageType


class CateAnalysis:
    def __init__(self, topk=1):
        self.coin_path = DictFilePath.coin_path
        self.stop_words_cn_path = DictFilePath.stop_words_cn_path
        self.stop_words_en_path = DictFilePath.stop_words_en_path
        self.idf_path = DictFilePath.idf_path
        self.cn_dict_path = DictFilePath.cn_dict_path
        self.en_dict_path = DictFilePath.en_dict_path
        self.synonym_path = DictFilePath.synonym_path
        self.cate_keyword_path = DictFilePath.cate_keyword_path
        self.model_path = DictFilePath.fasttext_model_path
        self.topk = topk
        self.min_score = 0.8
        self.max_idf = 6
        self.main_coin = {'BTC', 'ETH', 'NFT'}
        self.category_system, self.coin2cate, self.coins = self.load_coin()
        self.synonym = self.load_synonym()
        self.idf_dict = self.load_idf()
        self.content_cate_map, self.sec2frist, self.first_class, self.second_class, self.patterns_map = \
            self.load_cate_keyword()

        self.classifier = self.check_model(self.model_path)

        jieba.analyse.set_stop_words(self.stop_words_cn_path)
        jieba.analyse.set_stop_words(self.stop_words_en_path)
        jieba.analyse.set_idf_path(self.idf_path)
        jieba.load_userdict(self.cn_dict_path)
        jieba.load_userdict(self.en_dict_path)

    def load_coin(self):
        category_system = {}
        coins = set()
        coin2cate = {}
        for line in open(self.coin_path):
            line = line.strip().split('\t')
            if len(line) != 2: continue
            coin_line = line[1].split()  # .lower().split()
            category_system[line[0]] = coin_line
            for coin in coin_line:
                coins.add(coin)
                coin2cate[coin] = line[0]

        return category_system, coin2cate, coins

    def load_synonym(self):
        synonym = {}
        for line in open(self.synonym_path, 'r'):
            line = line.strip().split(';')
            if len(line) < 2: continue
            for syn in line[1:]:
                synonym[syn] = line[0]
        return synonym

    def load_idf(self):
        idf_dict = {}
        for line in open(self.idf_path, 'r').readlines():
            line = line.strip().split(' ')
            try:
                if len(line[0]) < 2:
                    continue
                idf_dict[line[0]] = float(line[1])
            except:
                print('error idf_dict line:', line)
        return idf_dict

    def load_cate_keyword(self):
        """
        加载类目
        加载类目关键词
        加载正则表达式
        """
        content_cate_map, sec2frist = {}, {}
        first_class, second_class = set(), {}
        patterns_map = {}
        tag = ''
        for line in open(self.cate_keyword_path, 'r'):
            if len(line.strip()) < 1: continue
            line = line.strip().split('\t')
            if len(line) == 1:
                tag = line[0]
            if len(line) != 2: continue
            if tag == "cate_map":
                sec_cate = line[1].split()
                content_cate_map[line[0]] = sec_cate
                for sec in sec_cate:
                    sec2frist[sec] = line[0]

            elif tag == "first_class_key":
                for sec in line[1].split():
                    first_class.add(line[0])
                    second_class[sec] = line[0]
            elif tag == "second_class_key":
                for sec in line[1].split():
                    second_class[sec] = line[0]
            elif tag == "patterns":
                patterns_map[line[0]] = line[1]

        for coin in self.coins:
            if coin not in self.main_coin:
                second_class[coin] = "其他代币"

        return content_cate_map, sec2frist, first_class, second_class, patterns_map

    def txt_clean(self, doc):
        doc = unicodedata.normalize('NFKC', doc)
        doc = re.sub(r'<[^>]+>', '', doc, 9999, re.S)
        doc = re.sub(r'<.*?>', '', doc, 9999, re.S)
        doc = re.sub(r'\r\n', '', doc, 9999, re.S)
        doc = re.sub(r'\n', '', doc, 9999, re.S)
        doc = re.sub(r'\t', '', doc, 9999, re.S)
        doc = re.sub(r'\r', '', doc, 9999, re.S)
        doc = re.sub(r'  ', '', doc, 9999, re.S)
        doc = re.sub(r'.td-.*?;}', '', doc, 9999, re.S)
        doc = re.sub(r'@media.*?}', '', doc, 9999, re.S)
        doc = re.sub(r'tdi.*?}', '', doc, 9999, re.S)
        doc = re.sub(r'  ', '', doc, 9999, re.S)
        doc = re.sub(r'; ;', '', doc, 9999, re.S)
        doc = re.sub(r'&#[0-9]*;', '', doc, 9999, re.S)
        return doc

    def cut_chinese(self, content):
        tokens = jieba.cut(content)
        return tokens

    def cut_english(self, content):
        tokens = nltk.word_tokenize(content)
        return tokens

    def get_word_list(self, doc):
        # 把句子按字分开，中文按字分，英文按单词，数字按空格
        regEx = re.compile('[\\W]+')  # 切分的规则是除单词，数字外的任意字符串
        res = re.compile(r"([\u4e00-\u9fa5])")  # 中文范围

        p1 = regEx.split(doc.lower())
        str_list = []
        for str in p1:
            if res.split(str) == None:
                str_list.append(str)
            else:
                ret = res.split(str)
                for ch in ret:
                    str_list.append(ch)
        list_word = [w for w in str_list if len(w.strip()) > 0]  # 去掉为空的字符
        return list_word

    def match_category_coin(self, content, language=LanguageType.Chinese_simplify):
        """
        资产类目预测
        """
        category_coin = []
        if language == LanguageType.Chinese_simplify:
            tokens = self.cut_chinese(content)
        else:
            tokens = self.cut_english(content)
        coin_num = {}
        for tok in tokens:
            tok = self.synonym.get(tok, tok)
            if tok not in self.coins:
                continue
            idf_num = coin_num.get(tok, [0, 0])
            if idf_num[0] == 0:
                idf_num[0] = 1
                idf_num[1] = self.idf_dict.get(tok.lower(), self.max_idf)
            else:
                idf_num[0] += 1
            coin_num[tok] = idf_num

        if coin_num:
            coin_score = {k: v[0] * v[1] for k, v in coin_num.items()}
            coin_score = sorted(coin_score.items(), key=lambda x: x[1], reverse=True)
            for i in range(min(self.topk, len(coin_score))):
                coin = coin_score[i][0]
                cate = self.coin2cate.get(coin, "其他币")
                category_coin.append([cate, coin])

        cate_coin = category_coin[0] if category_coin else ['', '']
        return cate_coin

    def match_category_content(self, content, language=LanguageType.Chinese_simplify, top1=True):
        """
        内容类目预测
        """
        category_content = []
        if language == LanguageType.Chinese_simplify:
            tokens = self.cut_chinese(content)
        else:
            tokens = self.cut_english(content)
        sec_score = {}
        for tok in tokens:
            tok = self.synonym.get(tok, tok)
            if tok not in self.second_class:
                continue
            sec_cate = self.second_class.get(tok)
            idf_score = sec_score.get(sec_cate, 0) + self.idf_dict.get(tok.lower(), self.max_idf)
            sec_score[sec_cate] = idf_score
        # 二级类目必须包含一级类目的关键词
        for frt in self.first_class:
            if frt not in sec_score:
                for sec in self.content_cate_map.get(frt, []):
                    sec_score[sec] = 0
            else:
                sec_score[frt] = 0

        if sec_score:
            sec_score = sorted(sec_score.items(), key=lambda x: x[1], reverse=True)
            for i in range(min(self.topk, len(sec_score))):
                if sec_score[i][1] <= 0: continue
                cate2 = sec_score[i][0]
                cate1 = self.sec2frist.get(cate2)
                category_content.append([cate1, cate2])

        if top1:
            cate_content = category_content[0] if category_content else ['', '']
        else:
            cate_content = category_content[:self.topk]
        return cate_content

    def cate_analysis(self, content, language=LanguageType.Chinese_simplify):
        """币种和内容分类主函数（非模型）"""
        category_coin = self.match_category_coin(content, language=language)
        category_content = self.match_category_content(content, language=language)
        return category_coin, category_content

    def check_model(self, model_path):
        if os.path.isfile(model_path):
            return fasttext.load_model(model_path)
        else:
            return None

    def model_test(self, content):
        """模型预测"""
        list_word = self.get_word_list(doc=content)
        text = ' '.join(list_word)
        labels = self.classifier.predict(text=text, k=self.topk)
        return labels

    def model_regular(self, content, language):
        """模型+规则预测"""
        res = {}
        regular_cate = self.match_category_content(content=content, language=language, top1=False)
        min_score = self.min_score / 2
        try:
            model_cate = self.model_test(content)
            for label, score in zip(model_cate[0], model_cate[1]):
                label = label.rsplit('__', 1)[1]
                res[label] = score
            min_score = self.min_score
        except:
            logging.info('load model error !')
        for i, line in enumerate(regular_cate):
            if len(line) != 2: continue
            cate = line[1]
            score = self.min_score - i * 0.2
            sc = res.get(cate, 0) + score
            res[cate] = sc
        res = sorted(res.items(), key=lambda x: x[1], reverse=True)
        if res and res[0][1] >= min_score:
            c2 = res[0][0]
            c1 = self.sec2frist.get(c2, '')
        else:
            c1 = c2 = ''
        return [c1, c2]


"""
ca = CateAnalysis(topk=2)
# udf_match_category_coin = Func.udf(lambda x, y: ca.match_category_coin(x, y), StructType(
#     [StructField("x", StringType(), True), StructField("y", StringType(), True)]))
# udf_match_category_content = Func.udf(lambda x, y: ca.match_category_content(x, y), StructType(
#     [StructField("x", StringType(), True), StructField("y", StringType(), True)]))
udf_match_category_coin_1 = Func.udf(lambda x, y: ca.match_category_coin(x, y)[0], StringType())
udf_match_category_coin_2 = Func.udf(lambda x, y: ca.match_category_coin(x, y)[1], StringType())
udf_match_category_content_1 = Func.udf(lambda x, y: ca.model_regular(x, y)[0], StringType())
udf_match_category_content_2 = Func.udf(lambda x, y: ca.model_regular(x, y)[1], StringType())
"""

if __name__ == '__main__':
    content = "btc news"
    language = LanguageType.English

    ca = CateAnalysis(topk=2)
    category_coin, category_content = ca.cate_analysis(content, language=language)
    print(category_coin, category_content)

    category_content = ca.model_regular(content, language=language)
    print(category_content)
