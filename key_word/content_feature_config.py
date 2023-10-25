import os
from collections import namedtuple

#base_path = os.path.dirname(__file__)
base_path = './'
print("base_path", base_path)


class ContentType:
    """
    内容类型： 1. 快讯 2 深度
    """
    unknown = 0
    flash = 1
    news = 2
    flash_weight = [0.25, 0.5, 0.15, 0.1]
    news_weight = [0.3, 0.3, 0.2, 0.2]
    ngram_n = 3
    max_tf_thd = {1: 500000, 2: 2000, 3: 1000, 4: 400}
    min_tf_thd = {1: 30, 2: 20, 3: 10, 4: 5}
    min_mpi = {1: 0, 2: 2.0, 3: 3.0, 4: 4.0}
    boundary_entropy_thd = 1.0
    pos_cn = {'n', 'nr', 'ns', 'nt', 'nz', 'vn', 'v', 'vd', 'vn', 'l', 'a', 'd'}
    pos_en = {'NN', 'NNS', 'NNP', 'NNPS', 'EX', 'FW', 'JJ', 'MD'}
    content_tag_num = 30
    max_idf = 6


class LanguageType:
    simplify_chinese = 1
    english = 3
    russian = 7


LanguageObject = namedtuple('LanguageObject', ['language_id', 'en_name', 'zh_name'])
lang_ZHCN = LanguageObject(1, "zh-CN", "中文简体")
lang_ZHTW = LanguageObject(2, "zh-TW", "中文繁体")
lang_ENUS = LanguageObject(3, "en-US", "英文")
lang_ZHHK = LanguageObject(5, "zh-HK", "香港")
lang_ENIN = LanguageObject(6, "en-IN", "印度英文")
lang_RURU = LanguageObject(7, "ru-RU", "俄语")
all_languages = (lang_ZHCN, lang_ZHTW, lang_ENUS, lang_ZHHK, lang_ENIN, lang_RURU)


class UseFilePath:
    stop_words_cn_path = os.path.join(base_path, 'dict/stopword_cn.txt')
    stop_words_en_path = os.path.join(base_path, 'dict/stopword_en.txt')
    idf_path = os.path.join(base_path, 'dict/idf_dict.txt')
    cn_dict_path = os.path.join(base_path, 'dict/cn_dict.txt')
    en_dict_path = os.path.join(base_path, 'dict/en_dict.txt')
    ngram_path = os.path.join(base_path, 'dict/ngram_dict.txt')


content_feature_config = {
    ContentType.flash: {
        "source_sql": "select f_id as ori_content_id, 1 as style_type, '' as title, f_content as ori_content, "
                      "f_language_id as language, f_issue_time as publish_time, '' as author, f_source as source "
                      "from huobi_global.ods_global_quotation_t_newsflash_da where dt='{}'"
    }
}
