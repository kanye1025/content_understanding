from frameworks.content_understanding.nlp_tools.word.word_utils import cut,pos_tag

text = 'Huobi Global数据监测显示,BTC跌破39500 USDT,现报39497.26 USDT,24H涨幅7.00%。'
#text = 'a program testor.py.py english for nltk,'
#text = '"Конфликты по всему миру, такие как вторжение на Украину и акты насилия, осуществляемые вооруженными группами, терроризирующими население Гаити, не являются законными средствами разрешения споров, поэтому государства-члены Конференции министров обороны стран Америки надеются на скорейшее мирное решение", - говорится в итоговой декларации участников конференции 21 страны континента, которая проходит в Бразилиа.'

tokens = cut(text,'zh')
for w in tokens:
    print(w)

for w in pos_tag(text= text,lang='zh'):
    print(w)




