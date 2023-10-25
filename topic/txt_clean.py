import sys 
sys.path.append("..") 
import pandas as pd
import html
import unicodedata
import re

from lda import *

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

if __name__=='__main__':
    # 返回的是一个DataFrame数据
    pd_reader = pd.read_csv("./frameworks/content_understanding/topic/item_feature.csv")
    #print(pd_reader['title'], pd_reader['content'], pd_reader['language'], pd_reader['style_type'])
    #print(pd_reader['tags'], pd_reader['entity_phrase'])

    docs = []
    for item in pd_reader.values:
        print('content, language, style_type:', txt_clean(item[2]), item[3], item[4])
        print('lda_topic: ', get_topic(txt_clean(item[2]), item[3], 1))
        #关键词抽取,关键短语抽取,关键句抽取
        break

    #print(get_summary(document))
