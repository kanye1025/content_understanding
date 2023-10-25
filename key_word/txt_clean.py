import sys 
sys.path.append("..") 
import pandas as pd
import html
import unicodedata
import re

from key_words2 import TfIdf

def txt_clean(doc):
    doc = html.unescape(doc)
    doc = unicodedata.normalize('NFKC', doc)
    doc = re.sub(r'<[^>]+>', '', doc, 9999, re.S)
    doc = re.sub(r'<.*?>', '', doc, 9999, re.S)
    doc = re.sub(r'\r\n', '.', doc, 9999, re.S)
    doc = re.sub(r'\n', '.', doc, 9999, re.S)
    doc = re.sub(r'\t', ' ', doc, 9999, re.S)
    doc = re.sub(r'\r', ' ', doc, 9999, re.S)
    doc = re.sub(r'  ', ' ', doc, 9999, re.S)
    doc = re.sub(r'.td-.*?;}', '', doc, 9999, re.S)
    doc = re.sub(r'@media.*?}', '', doc, 9999, re.S)
    doc = re.sub(r'tdi.*?}', '', doc, 9999, re.S)
    doc = re.sub(r'-', ' ', doc, 9999, re.S)      ##
    doc = re.sub(r'  ', '', doc, 9999, re.S)
    doc = re.sub(r'; ;', '', doc, 9999, re.S)
    doc = re.sub(r'&#[0-9]*;', '', doc, 9999, re.S)
    #doc = re.sub(r'  ', '', doc, 9999, re.S)
    return doc

def loadDataSet():
    dataset = [ ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],    # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'] ]
    classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表好，0代表不好
    return dataset, classVec
 

if __name__=='__main__':
    # 返回的是一个DataFrame数据
    pd_reader = pd.read_csv("./frameworks/content_understanding/key_word/item_feature.csv")
    #print(pd_reader['title'], pd_reader['content'], pd_reader['language'], pd_reader['style_type'])
    #print(pd_reader['tags'], pd_reader['entity_phrase'])

    docs = []
    for item in pd_reader.values:
        print('content, language, style_type:', txt_clean(item[2]), item[3], item[4])
        print('lp_tfidf: ', TfIdf().get_keywords_tfidf(txt_clean(item[2]), item[3], 1))
        #关键词抽取,关键短语抽取,关键句抽取
        break

    #print(get_summary(document))
