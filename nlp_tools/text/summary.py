from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor


def get_summary(text,sentence_count = 3,delimiter_list = ["。", "\n", ".", "【", "】", "?", "？", "!", "！"],lang='en'):
    '''
    get summary by using textrank to select the most important sentences  of the text.
    :param text: text to summary
    :param sentence_count:     max sentences count to return
    :param delimiter_list:     delimiters
    :param lang:               language ['cn','zh','ru','de',...]
    :return:                   [list]  sentences of the summary
    '''
    # Object of automatic summarization.
    auto_abstractor = AutoAbstractor()

    # Set tokenizer.
    auto_abstractor.tokenizable_doc = SimpleTokenizer()

    # Set delimiter for making a list of sentence.
    auto_abstractor.delimiter_list = delimiter_list

    # Object of abstracting and filtering document.
    abstractable_doc = TopNRankAbstractor()
    abstractable_doc.top_n = sentence_count
    # Summarize document.
    result_dict = auto_abstractor.summarize(text, abstractable_doc)

    ret = []
    # Output result.
    n_tail = len(auto_abstractor.delimiter_list) - 1
    for sentence in result_dict["summarize_result"]:
        ret.append(sentence[:-n_tail])
    return ret
