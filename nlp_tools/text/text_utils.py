
import re


def sentence_cut(text,sep = 'uncomplete'):
    '''

    :param text:分隔符 完整模式 sep = complete        u'。|；|？|！|\.|;|\?|!|\n'
                      非完整模式 sep = uncomplete    u'，|。|；|？|！|,|\.|;|\?|!|\n'
                      自定义模式 sep = 自定义正则分隔符
    :param sep:
    :return:
    '''
    if sep == 'complete':
        sep = u'。|；|？|！|\.|;|\?|!|\n'
    elif sep == 'uncomplete':
        sep = u'，|。|；|？|！|,|\.|;|\?|!|\n'

    for sentence in re.split(sep, text):
        if sentence:
            yield sentence


