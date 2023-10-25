from gensim import corpora, models
import jieba.posseg as jp
import jieba
import nltk

from pallas.frameworks.content_understanding.key_word.content_feature_config import ContentType, LanguageType
from pallas.frameworks.content_understanding.key_word.key_words2 import TfIdf


# 生成LDA模型
def LDA_model(words_list):
    # 构造词典
    # Dictionary()方法遍历所有的文本，为每个不重复的单词分配一个单独的整数ID，同时收集该单词出现次数以及相关的统计信息
    dictionary = corpora.Dictionary(words_list)
    print(dictionary)
    print('打印查看每个单词的id:')
    print(dictionary.token2id)  # 打印查看每个单词的id
 
    # 将dictionary转化为一个词袋
    # doc2bow()方法将dictionary转化为一个词袋。得到的结果corpus是一个向量的列表，向量的个数就是文档数。
    # 在每个文档向量中都包含一系列元组,元组的形式是（单词 ID，词频）
    corpus = [dictionary.doc2bow(words) for words in words_list]
    print('输出每个文档的向量:')
    print(corpus)  # 输出每个文档的向量
 
    # LDA主题模型
    # num_topics -- 必须，要生成的主题个数。
    # id2word    -- 必须，LdaModel类要求我们之前的dictionary把id都映射成为字符串。
    # passes     -- 可选，模型遍历语料库的次数。遍历的次数越多，模型越精确。但是对于非常大的语料库，遍历太多次会花费很长的时间。
    lda_model = models.ldamodel.LdaModel(corpus=corpus, num_topics=2, id2word=dictionary, passes=10)
 
    return lda_model
 
def get_topic(content, language, tag_score=0):
    kw = TfIdf()
    # 1=中文，3=英文, 7=俄文
    tokens = []
    try:
        if language == LanguageType.simplify_chinese:
            tokens = kw.preprocess_chinese(content)
        elif language == LanguageType.english:
            tokens = kw.preprocess_english(content)
        else:
            tokens = kw.preprocess_english_extend(content)

    except Exception as e:
        print(f'word segment error, {e}')


    # 获取分词后的文本列表
    words_list = [tokens]

    # 获取训练后的LDA模型
    lda_model = LDA_model(words_list)

    # 可以用 print_topic 和 print_topics 方法来查看主题
    # 打印所有主题，每个主题显示5个词
    topic_words = lda_model.print_topics(num_topics=2, num_words=5)
    #print('打印所有主题，每个主题显示5个词:')
    #print(topic_words)

    # 输出该主题的的词及其词的权重
    words_list = lda_model.show_topic(0, 5)
    #print('输出该主题的的词及其词的权重:')
    #print(words_list)
    return words_list
 
