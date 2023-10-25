import gensim
import numpy as np
from ..word.word_vec import get_word_vec
from ..word.word_utils import cut
from .tfidf import  get_tfidf_by_word
def get_text_vec(text,lang):
    '''
    暂时仅支持英文
    :param text:
    :param lang:
    :return:
    '''
    lang_set = set(['en', 'zh','ru'])
    assert lang in lang_set, f"lang must in {lang_set}"
    words_score = get_tfidf_by_word(text,lang)
    #tokens = cut(text,lang)
    sum_score = sum( [score for _,score in  words_score])


    words_vec = [get_word_vec(token,lang) * score for token,score in words_score ]
    return np.average(np.asarray(words_vec),axis = 0)/sum_score


