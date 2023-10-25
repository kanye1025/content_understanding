import os

base_path = os.path.dirname(os.path.dirname(__file__))
#base_path ='../nlp_tools/'
local_path = os.path.dirname(__file__)

class DictFilePath:
    coin_path = os.path.join(base_path, 'nlp_tools/dict/BI_intention.dat')
    stop_words_cn_path = os.path.join(base_path, 'nlp_tools/dict/stopword_cn.txt')
    stop_words_en_path = os.path.join(base_path, 'nlp_tools/dict/stopword_en.txt')
    idf_path = os.path.join(base_path, 'nlp_tools/dict/idf_dict.txt')
    cn_dict_path = os.path.join(base_path, 'nlp_tools/dict/cn_dict.txt')
    en_dict_path = os.path.join(base_path, 'nlp_tools/dict/en_dict.txt')
    synonym_coin_path = os.path.join(base_path, 'nlp_tools/dict/synonym_coin.txt')
    cate_keyword_path = os.path.join(base_path, 'nlp_tools/dict/cate_keyword.txt')

    synonym_path = os.path.join(base_path, 'nlp_tools/dict/synonym.txt')
    tag_dict_en_path = os.path.join(base_path, 'nlp_tools/dict/tag_dict_en.txt')
    tag_dict_cn_path = os.path.join(base_path, 'nlp_tools/dict/tag_dict_cn.txt')
    dynamic_coin_path = os.path.join(base_path, 'nlp_tools/dict/赛道标签数据（每周一更新).xlsx')

    lda_dictionary_path = os.path.join(local_path, 'model/lda_dictionary.dict')
    lda_model_path = os.path.join(local_path, 'model/lda.model')               # 待添加



class ContentType:
    pos_cn = {'n', 'nr', 'ns', 'nt', 'nz', 'vn', 'v', 'vd', 'vn', 'l', 'a', 'd'}
    pos_en = {'NN', 'NNS', 'NNP', 'NNPS', 'EX', 'FW', 'JJ', 'MD'}


class LanguageType:
    Chinese_simplify = 1
    English = 3
    Russian = 7
