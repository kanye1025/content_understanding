import jieba
import os

from .jieba_instantiation import posseg_cut,POSTokenizer

import nltk
jieba_ = jieba.Tokenizer()
file_dir = os.path.dirname(os.path.abspath(__file__))
jieba_.load_userdict(os.path.join(file_dir, '../dict', 'user_words_dict.txt'))
postok = POSTokenizer(jieba_)
def cut(text,lang):
    '''
    :param text:text to be cut
    :param lang: only in ['en','zh','su']
    :return: iterator of words
    '''
    if lang in['en','ru']:
        if lang == 'en':
            lang_ = 'english'
        elif lang == 'ru':
            lang_ = 'russian'
        return [w for w in nltk.word_tokenize(text,language = lang_)]
    elif lang == 'zh':
        return [w for w in jieba_.cut(text)]
    else:
        raise Exception('only [zh ,en,ru] are supported')

def cut_with_stop_words(sent,stop_words,lang ):
    ret = []
    for w in cut(sent,lang):
        if w not in stop_words:
            ret.append(w)
    return ret



def pos_tag(text = None,word_tokens = None,lang = 'en'):
    '''
    词性标注
    输入文本(text)或已分词list（word_tokens）,输出list(词,词性标注)
    :param text: 文本
    :param tokens:  已分词tokens，仅当text位空时生效
    :param lang: 仅支持 en，zh,ru
    :return:输出list(word,tag)
    '''

    assert lang in ('en','ru','zh') ,Exception('only [zh ,en,ru] are supported')
    if lang in ('en','ru'):
        import nltk
        if text:
            word_tokens = cut(text, lang=lang)
        if lang == 'en':
            lang_ = 'eng'
        elif lang=='ru':
            lang_ = 'rus'
        return nltk.pos_tag(word_tokens,lang = lang_)
    elif lang == 'zh':
        if not text:
            print(word_tokens)
            text = ''.join(word_tokens)

        return [(pair.word,pair.flag)for pair in posseg_cut(postok,text)]


