#/usr/bin/python
# coding:utf-8
import json
import math
import re

import jieba
import jieba.analyse
import jieba.posseg
import nltk
import pyspark.sql.functions as Func
from pyspark.sql.types import *

from pallas.frameworks.content_understanding.key_word.content_feature_config import ContentType, LanguageType
from pallas.frameworks.content_understanding.key_word.config import DictFilePath


# import logging
# logger = logging.getLogger(__name__)

"""
提取content的关键词向量，tf-idf
"""


class TfIdf:
    """
    load所有单词的idf；
    计算文本中单词的词频tf；
    计算每个单词的tf*idf；
    对重要性高的词重复采样
    """

    def __init__(self):
        self.idf_path = DictFilePath.idf_path
        self.stop_words_cn_path = DictFilePath.stop_words_cn_path
        self.stop_words_en_path = DictFilePath.stop_words_en_path
        self.cn_dict_path = DictFilePath.cn_dict_path
        self.en_dict_path = DictFilePath.en_dict_path
        self.pos_en = ContentType.pos_en
        self.pos_cn = ContentType.pos_cn
        self.topk = ContentType.content_tag_num
        self.max_idf = ContentType.max_idf
        self.punc = re.compile(r'[,/.=+_*&^…%￥$@!`~\\;:\"\'?<>(){}（）「」、？《》，。—]')
        self.idf_dict = self.load_idf()
        self.stop_words = self.load_stop()
        self.seg = jieba.analyse.set_stop_words(self.stop_words_cn_path)
        jieba.analyse.set_stop_words(self.stop_words_en_path)
        jieba.analyse.set_idf_path(self.idf_path)
        jieba.load_userdict(self.cn_dict_path)
        jieba.load_userdict(self.en_dict_path)

    def load_stop(self):
        stop_words = set()
        for line in open(self.stop_words_cn_path, 'r').readlines() + open(self.stop_words_en_path, 'r').readlines():
            line = line.strip()
            try:
                if len(line) < 2:
                    continue
                stop_words.add(line.lower())
            except:
                print(f'error idf_dict line: {line}')
        return stop_words

    def load_idf(self):
        idf_dict = {}
        for line in open(self.idf_path, 'r').readlines():
            line = line.strip().split(' ')
            try:
                if len(line[0]) < 2:
                    continue
                idf_dict[line[0]] = float(line[1])
            except:
                print(f'error idf_dict line: {line}')
        return idf_dict

    def preprocess_chinese(self, text):
        word_cn = []
        seg = jieba.posseg.cut(text)
        for i in seg:
            if (i.word not in self.stop_words) and (i.flag in self.pos_cn):  # 去停用词 + 词性筛选
                word_cn.append(i.word)
        return word_cn

    def preprocess_english(self, text):
        seg = nltk.word_tokenize(text)
        seg_pos = nltk.pos_tag(seg)
        word_en = []
        for i in seg_pos:
            word = i[0].lower()
            # 判断不在自定义词典里
            if (word not in self.stop_words) and (i[1] in self.pos_en):
                word_en.append(word)
        return word_en

    def preprocess_english_extend(self, text):
        seg = nltk.word_tokenize(text)
        seg_pos = nltk.pos_tag(seg)
        word_en = []
        for i in seg_pos:
            word = i[0].lower()
            # 判断不在自定义词典里
            if word not in self.stop_words:
                word_en.append(word)
        return word_en

    # jieba tf-idf获取文本top10关键词
    def extract_keywords(self, content, language=1, tag_score=0):
        # jieba.set_dictionary()
        # jieba.load_userdict()

        result_list = []  # 列表
        result_dict = {}  # 字典格式
        key_dict = {}
        try:
            jieba_result = jieba.analyse.extract_tags(content.lower(), topK=self.topk, withWeight=True, allowPOS=())
            for key_value in jieba_result:
                # 单字过滤、全数字过滤
                if (len(key_value[0]) < 2) or (key_value[0].isdigit()) or (re.search(self.punc, key_value[0])): continue
                key_dict[key_value[0]] = key_value[1]
            # 权重大的词重复采样
            avg = (sum(key_dict.values()) / len(key_dict) if len(key_dict) > 0 else 1) * 2
            for key, value in key_dict.items():
                result_dict[key] = value
                result_list.append(key)
                if value > avg:
                    result_list.append(key)
        except Exception as e:
            print(f'extract_keywords error, {e}')

        # logger.info("jieba obtain keyword with content length={}".format(len(content)))
        if tag_score == 0:
            return result_list
        else:
            return result_dict
 

    # tf-idf获取文本top30关键词
    def get_keywords_tfidf(self, content, language, tag_score=0):
        tf_dict, tfidf_dict = {}, {}
        # 1=中文，3=英文, 7=俄文
        tokens = []
        try:
            if language == LanguageType.simplify_chinese:
                tokens = self.preprocess_chinese(content)
            elif language == LanguageType.english:
                tokens = self.preprocess_english(content)
            else:
                tokens = self.preprocess_english_extend(content)

        except Exception as e:
            print(f'word segment error, {e}')

        num_score = 0
        for tok in tokens:
            # 单字过滤、全数字过滤
            if (len(tok) < 2) or (tok.isdigit()) or (re.search(self.punc, tok)): continue
            tf_dict[tok] = tf_dict.get(tok, 0) + 1
            num_score += 1
        num_score = math.log(num_score + 2, 2)

        result_list = []  # 列表
        result_dict = {}  # 字典格式
        key_dict = {}
        for tok, num in tf_dict.items():
            tfidf_dict[tok] = num ** 0.6 / num_score * self.idf_dict.get(tok, self.max_idf)

        tfidf_dict = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)
        
        print(self.topk)

        for item in tfidf_dict[:self.topk]:
            key_dict[item[0]] = item[1]  # * 0.1

        # 权重大的词重复采样
        avg = (sum(key_dict.values()) / len(key_dict) if len(key_dict) > 0 else 1) * 2
        for key, value in key_dict.items():
            result_list.append(key)
            result_dict[key] = value
            if value > avg:
                result_list.append(key)

        print("jieba obtain keyword with content length={}".format(len(content)))
        if tag_score == 0:
            return result_list
        else:
            return result_dict

    def operate_keywords(self, content_object, tag_score=0):
        handle_str = "{} {} {}".format(content_object.title, content_object.title, content_object.content)
        language = content_object.language
        keywords = self.get_keywords_tfidf(handle_str, language, tag_score)
        content_object.keywords = keywords
        return content_object


