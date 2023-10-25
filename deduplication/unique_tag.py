import logging
import os
import pickle
import re
import unicodedata

from bs4 import BeautifulSoup
from simhash import Simhash, SimhashIndex, long

from pallas.frameworks.content_understanding.deduplication.config.config_reader import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SimhashIndexV2(SimhashIndex):
    def __init__(self, objs, tolerance):
        super().__init__(objs, k=tolerance)

    def get_near_dups(self, simhash):
        """
        `simhash` is an instance of Simhash
        return a list of (obj_id, distance), which is in type of (str, int)
        """
        assert simhash.f == self.f

        ans = set()

        for key in self.get_keys(simhash):
            dups = self.bucket[key]
            self.log.debug('key:%s', key)
            if len(dups) > 200:
                self.log.warning('Big bucket found. key:%s, len:%s', key, len(dups))

            for dup in dups:
                sim2, obj_id = dup.split(',', 1)
                sim2 = Simhash(long(sim2, 16), self.f)

                d = simhash.distance(sim2)
                if d <= self.k:
                    ans.add((obj_id, d))
        return list(ans)


class UniqueTag:
    def __init__(self, docs, text_type):
        """
        `docs` is a list of(obj_id, simhash)
        """
        if text_type == "long":
            tolerance = config['simhash']['long_tolerance']
        else:
            tolerance = config['simhash']['short_tolerance']
        self.len_limit = config['simhash']['len_limit']
        self.re_max_len = config['simhash']['re_max_len']
        self.simhash_index_file_path = os.path.join(os.path.dirname(__file__), config['simhash']['save_path'])
        if os.path.exists(self.simhash_index_file_path):
            self.load_index()
        else:
            self.index = SimhashIndexV2(docs, tolerance)

    def get_doc_similar_id(self, doc_id, doc):
        """如果查询到相似文章，则sim_id设置为相似文章的id，如果没有查询到相似文章，则用自己的id当作sim_id,同时将文章添加到索引中"""
        doc_simhash = self.get_simhash(doc)
        doc_similar_id_list = self.get_sim_doc(doc_simhash)
        if doc_similar_id_list:
            print("重复文章")
            print(doc_similar_id_list)
            return doc_similar_id_list[0][0]
        else:
            self.add_to_index(doc_id, self.get_simhash(doc))
            return doc_id

    def get_simhash(self, doc):
        clean_doc = self.txt_clean(doc)[:self.len_limit]
        doc_simhash = Simhash(clean_doc)
        return doc_simhash

    def get_sim_doc(self, simhash):
        doc_sim_list = self.index.get_near_dups(simhash)
        return sorted(doc_sim_list, key=lambda x: x[1])

    def add_to_index(self, obj_id, simhash):
        self.index.add(obj_id, simhash)

    def delete_from_index(self, obj_id, simhash):
        self.index.delete(obj_id, simhash)

    def txt_clean(self, doc):
        clean_doc = doc.strip()
        soup = BeautifulSoup(clean_doc, 'html.parser')
        clean_doc = soup.get_text()
        clean_doc = unicodedata.normalize('NFKC', clean_doc)
        clean_doc = re.sub(r'\t', ' ', clean_doc, self.re_max_len)
        clean_doc = re.sub(r'\r', ' ', clean_doc, self.re_max_len)
        clean_doc = re.sub(r'.td-.*?;}', '', clean_doc, self.re_max_len)
        clean_doc = re.sub(r'@media.*?}', '', clean_doc, self.re_max_len)
        clean_doc = re.sub(r'tdi.*?}', '', clean_doc, self.re_max_len)
        return clean_doc

    def save_index(self):
        with open(self.simhash_index_file_path, 'wb') as f:
            pickle.dump(self.index, f)

    def load_index(self):
        with open(self.simhash_index_file_path, 'rb') as f:
            self.index = pickle.load(f)


class TextDeduplication:
    def __init__(self, long_docs, short_docs):
        self.l_text_uni_tag = UniqueTag(long_docs, "long")
        self.s_text_uni_tag = UniqueTag(short_docs, "short")

    def get_sim_id(self, doc_type, doc_id, doc_content):
        if doc_type == "long":
            sim_id = self.l_text_uni_tag.get_doc_similar_id(doc_id, doc_content)
        else:
            sim_id = self.s_text_uni_tag.get_doc_similar_id(doc_id, doc_content)
        return sim_id
