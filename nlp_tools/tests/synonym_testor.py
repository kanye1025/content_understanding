from frameworks.content_understanding.nlp_tools.word.word_vec import get_synonyms,get_synonyms_by_words

'''
#words = ['fall','drop','decline','depreciate','fall','downturn','plummet','stagnate','recession']
#words = ['rise','emergence','increasethe','growth','dominate','undepreciated','upswing']
words = ['no','not','neither','never','nor']
for w in get_synonyms_by_words(words,20,'en'):
    print(w)

'''
'''
#words = ['上涨','增长','涨','大涨','走高','冲高','走强','翻绿']
#words = ['下跌','走低','跌超','下挫','走弱','大跌']
words = ['不','不是','没有','没','并未','并非']
for w in get_synonyms_by_words(words,20,'zh'):
    print(w)
'''

words = ['BTC']
#words = ['рост','подъемный','возрастание']
#words = ['снижение','уменьшение','убывание','ослабление','упасть','падать']
for w in get_synonyms_by_words(words,20,'ru'):
    print(w)