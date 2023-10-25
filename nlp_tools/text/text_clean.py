import re
import unicodedata

from bs4 import BeautifulSoup


def txt_clean(doc):
    clean_doc = doc.strip()
    soup = BeautifulSoup(clean_doc, 'html.parser')
    clean_doc = soup.get_text()
    clean_doc = unicodedata.normalize('NFKC', clean_doc)
    clean_doc = re.sub(r'\t', ' ', clean_doc, 9999)
    clean_doc = re.sub(r'\r', ' ', clean_doc, 9999)
    clean_doc = re.sub(r'.td-.*?;}', '', clean_doc, 9999)
    clean_doc = re.sub(r'@media.*?}', '', clean_doc, 9999)
    clean_doc = re.sub(r'tdi.*?}', '', clean_doc, 9999)
    return clean_doc