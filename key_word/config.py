import os

base_path = os.path.dirname(os.path.dirname(__file__))

class DictFilePath:
    coin_path = os.path.join(base_path, 'nlp_tools/dict/BI_intention.dat')
    stop_words_cn_path = os.path.join(base_path, 'nlp_tools/dict/stopword_cn.txt')
    stop_words_en_path = os.path.join(base_path, 'nlp_tools/dict/stopword_en.txt')
    idf_path = os.path.join(base_path, 'nlp_tools/dict/idf_dict.txt')
    cn_dict_path = os.path.join(base_path, 'nlp_tools/dict/cn_dict.txt')
    en_dict_path = os.path.join(base_path, 'nlp_tools/dict/en_dict.txt')
    synonym_coin_path = os.path.join(base_path, 'nlp_tools/dict/synonym_coin.txt')
    cate_keyword_path = os.path.join(base_path, 'nlp_tools/dict/cate_keyword.txt')
