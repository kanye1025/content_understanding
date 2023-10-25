
from frameworks.content_understanding.nlp_tools.text.text_utils import *


#text = '狭义的汉语是指中文标准普通语言，也称普通话。中文标准普通语言是中华人民共和国通用语言，且为国际通用语言之一。广义的汉语是指汉语族，属汉藏语系。准确的说是以汉文字为唯一文字的一切语支的统称。历史沿革 上古 汉语属于汉藏语系。'
#text = "The academy is the designer and builder of the Long March 5B, the most powerful Chinese rocket when it comes to carrying capacity for low-Earth orbit. The rocket is central to China's space station program because it is now the only Chinese launch vehicle capable of carrying large space station parts into orbit."
text = "Депутат городского совета Новосибирска Хельга Пирогова покинула Россию после того, как в отношении неё было возбуждено уголовное дело о фейках о российской армии. Об этом информирует РИА Новости со ссылкой на её коллегу, депутата Светлану Каверзину."
for sentence in sentence_cut(text):
    print(sentence)