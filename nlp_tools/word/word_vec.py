import os.path

import fasttext
from collections import defaultdict
from ..utils.decorator import lazy_property

from ....util.model_s3 import check_and_make_sure_models



class __models__:

    @lazy_property
    def ft_models_dir(self):
        ft_model_version = '1.0.0'
        s3_dir = 's3://global-global-base-iea-item-store/content_understanding/word_vec/'
        local_dir = os.path.join(os.environ['HOME'], 'models/word_vec')
        file_list = ['huobi_en.ft', 'huobi_en.wd', 'huobi_ru.ft', 'huobi_ru.wd', 'huobi_zh.ft', 'huobi_zh.wd']
        return check_and_make_sure_models(s3_dir, local_dir, ft_model_version, file_list)

    @lazy_property
    def ft_model_en(self):

        ft_model_en_path = os.path.join(self.ft_models_dir,'huobi_en.ft')
        return fasttext.load_model(ft_model_en_path)

    @lazy_property
    def ft_model_zh(self):
        ft_model_zh_path = os.path.join(self.ft_models_dir,'huobi_zh.ft')
        return fasttext.load_model(ft_model_zh_path)

    @lazy_property
    def ft_model_ru(self):
        ft_model_ru_path = os.path.join(self.ft_models_dir,'huobi_ru.ft')
        return fasttext.load_model(ft_model_ru_path)


def __get_ft_model__(lang):
    lang_set = set(['en', 'zh', 'ru'])
    assert lang in lang_set, f"lang must in {lang_set}"
    if lang == 'en':
        model = __models__.ft_model_en
    elif lang=='zh':
        model = __models__.ft_model_zh
    elif lang=='ru':
        model = __models__.ft_model_ru
    return model
def get_word_vec(word,lang):
    '''
    获取词向量，目前仅支持zh ,en ,ru
    :param word:
    :param lang:
    :return:
    '''
    if lang in ('en','ru'):
        word = word.lower()
    model = __get_ft_model__(lang)
    try:
        vec = model.get_word_vector(word)
        return vec
    except:
        return None


def get_synonyms(word,k,lang):
    '''
    获取近义词，目前仅支持zh ,en ,ru
    :param word:
    :param k:
    :param lang:
    :return:
    '''
    if lang in ('en','ru'):
        word = word.lower()
    model = __get_ft_model__(lang)
    return model.get_nearest_neighbors(word,k)


def get_synonyms_by_words(words,k,lang):
    ret = defaultdict(float)
    for word in words:
        for score ,synonym in get_synonyms(word,k,lang):
            if synonym not in words:
                ret[synonym] +=score
    l = [(k,v)for k,v in ret.items()]
    l = sorted(l,key = lambda x:x[1],reverse=True)
    return list(l)[:k]
