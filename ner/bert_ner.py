# -*- encoding: utf-8 -*-

import jieba
import jieba.posseg
import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import BertTokenizer, logging
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from pallas.frameworks.content_understanding.key_word.content_feature_config import ContentType, LanguageType
from pallas.frameworks.util.model_s3 import check_and_make_sure_models


class BertNer:
    def __init__(self):
        self.tag_type_cn = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
        self.tag_type_en = ['O', 'art-broadcastprogram', 'art-film', 'art-music', 'art-other', 'art-painting', 'art-writtenart', 'building-airport', 'building-hospital', 'building-hotel', 'building-library', 'building-other', 'building-restaurant', 'building-sportsfacility', 'building-theater', 'event-attack/battle/war/militaryconflict', 'event-disaster', 'event-election', 'event-other', 'event-protest', 'event-sportsevent', 'location-bodiesofwater', 'location-GPE', 'location-island', 'location-mountain', 'location-other', 'location-park', 'location-road/railway/highway/transit', 'organization-company', 'organization-education', 'organization-government/governmentagency', 'organization-media/newspaper', 'organization-other', 'organization-politicalparty', 'organization-religion', 'organization-showorganization', 'organization-sportsleague', 'organization-sportsteam', 'other-astronomything', 'other-award', 'other-biologything', 'other-chemicalthing', 'other-currency', 'other-disease', 'other-educationaldegree', 'other-god', 'other-language', 'other-law', 'other-livingthing', 'other-medical', 'person-actor', 'person-artist/author', 'person-athlete', 'person-director', 'person-other', 'person-politician', 'person-scholar', 'person-soldier', 'product-airplane', 'product-car', 'product-food', 'product-game', 'product-other', 'product-ship', 'product-software', 'product-train', 'product-weapon']
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.s3_dir = 's3://global-global-base-iea-item-store/content_understanding/ner_model/'
        self.local_dir ='/usr/local/models/ner_model/'
        self.file_list = ['bert-ner-en.pt', 'bert-ner.pt']
        check_and_make_sure_models(self.s3_dir, self.local_dir, '1658829299', self.file_list)

        self.model_cn = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=7)
        self.model_cn.to(self.device)
        self.model_cn = torch.load('/usr/local/models/ner_model/1658829299/bert-ner.pt')
        
        self.model_en = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=67)
        self.model_en.to(self.device)
        self.model_en = torch.load('/usr/local/models/ner_model/1658829299/bert-ner-en.pt')


    def predcit_cn(self, s):
        ################## 4. 定义模型
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        item = tokenizer([s], truncation=True, padding='longest', max_length=64)  # 加一个list
        with torch.no_grad():
            input_ids = torch.tensor(item['input_ids']).to(self.device).reshape(1, -1)
            attention_mask = torch.tensor(item['attention_mask']).to(self.device).reshape(1, -1)
            labels = torch.tensor([0] * attention_mask.shape[1]).to(self.device).reshape(1, -1)

            outputs = self.model_cn(input_ids, attention_mask, labels)
            outputs = outputs[0].data.cpu().numpy()

        outputs = outputs[0].argmax(1)[1:-1]
        print('outputs',outputs)
        ner_result = []
        ner_flag = ''

        for o, c in zip(outputs, s):
            # 0 就是 O，没有含义
            if o == 0 and ner_result == '':
                continue
            #
            # elif o == 0 and ner_result != '':
            #     if ner_flag == 'O':
            #         print('机构：', ner_result)
            #     if ner_flag == 'P':
            #         print('人名：', ner_result)
            #     if ner_flag == 'L':
            #         print('位置：', ner_result)
            elif o != 0:
                ner_flag += c
        
        ner_result.extend([item.word for item in jieba.posseg.cut(ner_flag)])
        
        return ner_result


    def predcit_en(self, s):
        ################## 4. 定义模型
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        item = tokenizer([s], truncation=True, padding='longest', max_length=64)  # 加一个list
        with torch.no_grad():
            input_ids = torch.tensor(item['input_ids']).to(self.device).reshape(1, -1)
            attention_mask = torch.tensor(item['attention_mask']).to(self.device).reshape(1, -1)
            labels = torch.tensor([0] * attention_mask.shape[1]).to(self.device).reshape(1, -1)

            outputs = self.model_en(input_ids, attention_mask, labels)
            outputs = outputs[0].data.cpu().numpy()

        outputs = outputs[0].argmax(1)[1:-1]
        print('outputs',outputs)
        ner_result = []
        ner_flag = ''

        for o, c in zip(outputs, s.strip().split()):
            # 0 就是 O，没有含义
            if o == 0 and ner_result == '':
                continue
            #
            # elif o == 0 and ner_result != '':
            #     if ner_flag == 'or':
            #         print('机构：', ner_result)
            #     if ner_flag == 'pe':
            #         print('人名：', ner_result)
            #     if ner_flag == 'lo':
            #         print('位置：', ner_result)
            elif o != 0:
                ner_result.append(c)
        return ner_result

    def bert_ner(self, content, language, tag_score=0):
        logging.set_verbosity_warning()
        logging.set_verbosity_error()

        # 1=中文，3=英文, 7=俄文
        tokens = []
        try:
            if language == LanguageType.simplify_chinese:
                tokens = self.predcit_cn(content)
            elif language == LanguageType.english:
                tokens = self.predcit_en(content)
            else:
                tokens = self.predcit_en(content)

        except Exception as e:
            print('word segment error, {e}')

        return tokens

if __name__ == '__main__':
    bn = BertNer()
    content = '整个华盛顿已笼罩在一片夜色之中，一个电话从美国总统府白宫打到了菲律宾总统府马拉卡南宫。'
    bn.bert_ner(content, 1, 1)
    content = 'following his retirement from cricket , pitman held a coaching and administrative post at rydal penrhos school in wales .'
    bn.bert_ner(content, 3, 1)

