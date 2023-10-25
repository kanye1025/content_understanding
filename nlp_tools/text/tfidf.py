from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import joblib
from ...toolkit.utils.data_file import DataFile
from ....util.model_s3 import check_and_make_sure_models
import os
from ...toolkit.utils.cached import cached
from ...toolkit.utils.decorator import callonce

model_dir_s3 = 's3://global-global-base-iea-item-store/content_understanding/tfidf/'
model_dir_local = os.path.join(os.environ['HOME'],'models/tfidf/')
file_list = ['en.tfidf','zh.tfidf','ru.tfidf','huobi_en.wd','huobi_zh.wd','huobi_ru.wd']

model_dir_local_version = check_and_make_sure_models(model_dir_s3,model_dir_local,version = '1.0.0',file_list = file_list)
print('update tfidf_model')



@cached()
def get_word_vocab(lang):
    vocab_path = os.path.join(model_dir_local_version,f'huobi_{lang}.wd')
    vocab_list = list(DataFile.load_string_list(vocab_path))
    vocab_dict = {w: i for i, w in enumerate(vocab_list)}
    return vocab_list,vocab_dict


@cached()
def get_vectorizer(lang):
    vocab_list, vocab_dict = get_word_vocab(lang)
    vectorizer = CountVectorizer(vocabulary=vocab_dict,lowercase = False)
    return vectorizer

@cached()
def get_tfidf_model(lang):
    model_path = os.path.join(model_dir_local_version,f'{lang}.tfidf')
    tf_idf_transformer = joblib.load(model_path)
    return tf_idf_transformer


def get_tfidf_vec(text,lang):
    vectorizer = get_vectorizer(lang)
    tf_idf_transformer = get_tfidf_model(lang)
    x = vectorizer.transform([text])
    x = tf_idf_transformer.transform(x)

    return x.toarray()[0]



def get_tfidf_by_word(text,lang,topK = None):
    vectorizer = get_vectorizer(lang)
    tf_idf_transformer = get_tfidf_model(lang)
    vocab_list,_ = get_word_vocab(lang)
    x = vectorizer.transform([text])
    x = tf_idf_transformer.transform(x)
    x = x.tocoo()
    x = sorted(zip(x.col, x.data), key=lambda i: i[1], reverse=True)
    if topK:
        x = x[:topK]
    return [(vocab_list[index],score )for (index,score) in x]




