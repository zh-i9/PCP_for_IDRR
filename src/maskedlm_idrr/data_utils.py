import re
import os
import sys
import json
import random
import torch
import pytorch_lightning as pl

from tqdm import tqdm
from pprint import pprint
from dataclasses import dataclass
from collections import Counter
from transformers import BertTokenizerFast, RobertaTokenizerFast
from q_snippets.object import Config, print_config, print_string
from q_snippets.data import BaseData, sequence_padding, save_json
from torch.utils.data import Dataset, DataLoader, random_split
from data.causal_relation.pdtb2.pdtb2 import CorpusReader

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class InputSample:  # 原始数据
    arg1: str
    arg2: str
    connective: str
    sense: str


@dataclass
class InputFeature:  # 处理后的特征数据
    input_ids: list
    attention_mask: list
    # token_type_ids: list
    mask_token_id: int
    label: int


@dataclass
class InputBatch(BaseData):  # 对特征数据进行batch处理
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    # token_type_ids: torch.Tensor
    mask_token_ids: torch.Tensor
    labels: torch.Tensor

    def __len__(self):
        return self.input_ids.size(0)

    def info(self):
        size = {k: v.size() if type(v) is torch.Tensor else len(v)
                for k, v in self.__dict__.items()}
        print(size)


class LabelMapping(object):
    def __init__(self, mapping_path):
        with open(mapping_path, "r") as f:
            self.label2words:dict = json.load(f)
        self.words2label = self.get_words2label(self.label2words)
        labels = sorted(list(set(self.words2label.values())))
        self.label2id = self.get_label2id(labels)
        self.id2label = {id: label for label, id in self.label2id.items()}

    def get_words2label(self, label2words):
        words2label = {}
        for label, words in label2words.items():
            # if label.startswith("Contingency"):  # causality
            #     for word in words:
            #         words2label[word] = label
            # else:
            for word in words:
                words2label[word] = label.split(".")[0]  # pdtb2 top-level
                # words2label[word] = ".".join(label.split(".")[:2])  # pdtb2 second-level
                # words2label[word] = label  # cross-level
        return words2label
    
    def get_label2id(self, labels):
        label2id = {"Other": 0}
        for idx, label in enumerate(labels):
            label2id[label] = idx+1  # 1~n
        return label2id


class CoNll16Dataset(Dataset):
    def __init__(self, config, tokenizer, dataset_mode) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.mode = dataset_mode
        self.label_mapping = LabelMapping(config.mapping_path)
        self.label2words = self.label_mapping.label2words
        self.words2label = self.label_mapping.words2label
        self.samples, self.features = self._handle_cache()

    def _handle_cache(self):
        """
            核心是 self.load_data 加载并处理数据，返回原始数据和处理后的特征数据
            需要注意缓存文件与 self.config.cache_dir  self.mode 有关
        Returns:
            samples, features
        """
        os.makedirs(self.config.cache_dir, exist_ok=True)               # 确保缓存文件夹存在
        file = os.path.join(self.config.cache_dir, f"{self.mode}.pt")   # 获得缓存文件的路径
        # 如果已经存在，且没有强制重新生成，则从缓存读取
        if os.path.exists(file) and not self.config.force_reload:
            samples, features = torch.load(file)
            print(f"{len(samples), len(features)} samples, features loaded from {file}")
            return samples, features
        else:
            samples, features = self.load_data()                        # 读取并处理数据
            torch.save((samples, features), file)                       # 生成缓存文件
            return samples, features

    def load_data(self):
        samples, features = [], []
        if self.mode == 'train':
            path = self.config.train_path
        elif self.mode == 'val':
            path = self.config.val_path
        elif self.mode == 'test':
            path = self.config.test_path
        with open(path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        
        sense_list = []
        for dic in tqdm(all_data):
            arg1 = dic["Arg1"]
            arg2 = dic["Arg2"]
            
            sense = dic["Sense"]
            if sense not in self.label2words.keys():
                continue
            
            connective = dic["Connective"]
            if connective in self.label2words[sense]:
                mask_word = connective
            else:  # 对于其他连接词用频率最高的连接词进行替换
                mask_word = self.label2words[sense][0]
            label = self.tokenizer.convert_tokens_to_ids(mask_word)

            # prompt = "[CLS] " + arg1 + " [MASK] " + arg2 + ". [SEP]"  # bert prompt
            # prompt = "<s> " + arg1.capitalize() + " <mask> " + arg2.lower() + ". </s>"  # roberta prompt1
            # prompt = "<s> " + arg1.capitalize() + ". That's <mask> " + arg2.lower() + ". </s>"  # roberta prompt2
            prompt3 = "Arg1: " + arg1.capitalize() + ". Arg2: " + arg2.capitalize() + ". The conjunction between Arg1 and Arg2 is <mask>."  # roberta prompt3
            prompt4 = "<s>Arg1: " + arg1.capitalize() + ". Arg2: " + arg2.capitalize() + ". The conjunction between Arg1 and Arg2 is <mask>.</s>"  # roberta prompt4
            inputs = self.tokenizer(
                prompt4,
                max_length=512,
                truncation=True,
                return_offsets_mapping=True
            )

            if self.tokenizer.mask_token_id not in inputs.input_ids: # 如果arg1超过512限制则舍弃
                continue
            mask_token_id = inputs.input_ids.index(self.tokenizer.mask_token_id)
            
            if self.mode in ["train", "val", "test"]:
                sense_list.append(sense)
                samples.append(InputSample(
                    arg1=arg1, arg2=arg2, 
                    connective=connective, 
                    sense=sense
                ))
                features.append(InputFeature(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    # token_type_ids=inputs.token_type_ids,
                    mask_token_id=mask_token_id,
                    label=label
                ))
            else:
                pass
        
        print(dict(Counter(sense_list)))
        print(f"{str(len(samples))} samples, {str(len(features))} features in total for {self.mode}")

        return samples, features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    @staticmethod
    def collate_fn(batch):
        test = batch[0].label is None
        max_len = min(max([len(x.input_ids) for x in batch]), 512)

        # 将batch_size条feature转换成Tensor
        input_ids = torch.tensor(sequence_padding([x.input_ids for x in batch], length=max_len))
        attention_mask = torch.tensor(sequence_padding([x.attention_mask for x in batch], length=max_len))
        # token_type_ids = torch.tensor(sequence_padding([x.token_type_ids for x in batch], length=max_len))
        mask_token_ids = torch.tensor([x.mask_token_id for x in batch])
        labels = torch.tensor([x.label for x in batch]) if not test else None
        batch = InputBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            mask_token_ids=mask_token_ids,
            labels=labels
        )
        return batch


class PDTB2Dataset(Dataset):
    def __init__(self, config, tokenizer, dataset_mode) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.mode = dataset_mode
        self.label_mapping = LabelMapping(config.mapping_path)
        self.label2words = self.label_mapping.label2words
        self.words2label = self.label_mapping.words2label
        self.samples, self.features = self._handle_cache()

    def _handle_cache(self):
        """
            核心是 self.load_data 加载并处理数据，返回原始数据和处理后的特征数据
            需要注意缓存文件与 self.config.cache_dir  self.mode 有关
        Returns:
            samples, features
        """
        os.makedirs(self.config.cache_dir, exist_ok=True)               # 确保缓存文件夹存在
        file = os.path.join(self.config.cache_dir, f"{self.mode}.pt")   # 获得缓存文件的路径
        # 如果已经存在，且没有强制重新生成，则从缓存读取
        if os.path.exists(file) and not self.config.force_reload:
            samples, features = torch.load(file)
            print(f"{len(samples), len(features)} samples, features loaded from {file}")
            return samples, features
        else:
            samples, features = self.load_data()                        # 读取并处理数据
            torch.save((samples, features), file)                       # 生成缓存文件
            return samples, features

    def load_data(self):
        samples, features = [], []
        if self.mode == 'train':
            path = self.config.train_path
        elif self.mode == 'val':
            path = self.config.val_path
        elif self.mode == 'test':
            path = self.config.test_path
        
        pdtb = CorpusReader(path)
        sense_list = []
        for datum in pdtb.iter_data(display_progress=True):
            if datum.Relation == "Implicit":
                # continue
                arg1:str = datum.Arg1_RawText.strip()
                arg2:str = datum.Arg2_RawText.strip()      
                sense = datum.ConnHeadSemClass1
                connective = datum.Conn1.strip()

                if sense not in self.label2words.keys():
                    continue
                if "Ġ"+connective in self.label2words[sense]:
                    mask_word = "Ġ"+connective  # roberta
                    # mask_word = connective  # bert
                else:  # 对于其他连接词用频率最高的连接词进行替换
                    mask_word = self.label2words[sense][0]  # roberta
                    # mask_word = self.label2words[sense][0][1:]  # bert
                label = self.tokenizer.convert_tokens_to_ids(mask_word)

                # prompt = "[CLS] " + arg1.strip().capitalize() + " [MASK] " + arg1.strip().lower() + ". [SEP]"  # bert prompt1
                # prompt = "[CLS] Arg1: " + arg1.strip().capitalize() + ". Arg2: " + arg2.strip().capitalize() + ". [SEP] The conjunction between Arg1 and Arg2 is [MASK]. [SEP]"  # bert prompt2

                roberta_prompt1 = arg1.capitalize() + " <mask> " + arg2.lower() + "."  # roberta prompt1
                roberta_prompt2 = arg1.capitalize() + ". That's <mask> " + arg2.lower() + "."  # roberta prompt2
                roberta_prompt3 = "Arg1: " + arg1.capitalize() + ". Arg2: " + arg2.capitalize() + ".</s></s>The conjunction between Arg1 and Arg2 is <mask>."  # roberta prompt3
                roberta_prompt4 = "Arg1: " + arg1.capitalize() + ". Arg2: " + arg2.capitalize() + ".</s></s>The connective between Arg1 and Arg2 is <mask>."  # roberta prompt4
                roberta_prompt5 = "Arg1: " + arg1.capitalize() + ". Arg2: " + arg2.capitalize() + ". The conjunction between Arg1 and Arg2 is <mask>."  # roberta prompt5
                roberta_prompt6 = "Arg1: " + arg1.capitalize() + ". Arg2: " + arg2.capitalize() + ". The connective between Arg1 and Arg2 is <mask>."  # roberta prompt6
                roberta_prompt7 = "<s>Arg1: " + arg1.capitalize() + ". Arg2: " + arg2.capitalize() + ". The conjunction between Arg1 and Arg2 is <mask>.</s>"  # roberta prompt7
                roberta_prompt8 = "<s>Arg1: " + arg1.capitalize() + ". Arg2: " + arg2.capitalize() + ". The connective between Arg1 and Arg2 is <mask>.</s>"  # roberta prompt8
                roberta_prompt9 = "<s>Arg1: " + arg1.capitalize() + ". Arg2: " + arg2.capitalize() + ". In summary, the discourse relation between Arg1 and Arg2 is <mask>.</s>"  # roberta prompt9
                roberta_prompt10 = "Arg1: " + arg1.capitalize() + ". Arg2: " + arg2.capitalize() + ". In summary, the discourse relation between Arg1 and Arg2 is <mask>."  # roberta prompt10

                inputs = self.tokenizer(roberta_prompt5, max_length=512, truncation=True)

                # if self.tokenizer.mask_token_id not in inputs.input_ids:  # 如果arg1超过512限制则舍弃
                #     continue
                mask_token_id = inputs.input_ids.index(self.tokenizer.mask_token_id)

                if self.mode in ["train", "val", "test"]:
                    # sense_list.append(sense)
                    samples.append(InputSample(
                        arg1=arg1, arg2=arg2, 
                        connective=connective,
                        sense=sense
                    ))
                    features.append(InputFeature(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        # token_type_ids=inputs.token_type_ids,
                        mask_token_id=mask_token_id,
                        label=label
                    ))
            
            if datum.Relation == "Explicit":
                continue
                # if datum.arg1_contains_arg2() or datum.arg2_contains_arg1():
                #     continue
                arg1:str = datum.Arg1_RawText.strip()
                arg2:str = datum.Arg2_RawText.strip()      
                sense:str = datum.ConnHeadSemClass1
                connective:str = datum.Connective_RawText.strip()

                if sense not in self.label2words.keys():
                    continue
                mask_word = self.label2words[sense][0]  # roberta
                # mask_word = self.label2words[sense][0][1:]  # bert
                label = self.tokenizer.convert_tokens_to_ids(mask_word)

                roberta_prompt1 = "Arg1: " + arg1.capitalize() + ". Arg2: " + arg2.capitalize() + \
                                  ". The connective between Arg1 and Arg2 is " + connective.lower() + \
                                  ". In summary, the discourse relation between Arg1 and Arg2 is <mask>."
                                  #  ". In summary, Arg2 is the <mask> of Arg1."

                inputs = self.tokenizer(roberta_prompt1, max_length=512, truncation=True)
                if self.tokenizer.mask_token_id not in inputs.input_ids:  # 如果prompt的总长度超过512限制则舍弃
                    continue
                mask_token_id = inputs.input_ids.index(self.tokenizer.mask_token_id)
                
                if self.mode in ["train", "val", "test"]:
                    sense_list.append(sense)
                    samples.append(InputSample(
                        arg1=arg1, arg2=arg2, 
                        connective=connective, 
                        sense=sense
                    ))
                    features.append(InputFeature(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        # token_type_ids=inputs.token_type_ids,
                        mask_token_id=mask_token_id,
                        label=label
                    ))
            
        # print(dict(Counter(sense_list)))
        print(f"{str(len(samples))} samples, {str(len(features))} features in total for {self.mode}")

        return samples, features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

    @staticmethod
    def collate_fn(batch):
        test = batch[0].label is None
        max_len = min(max([len(x.input_ids) for x in batch]), 512)

        # 将batch_size条feature转换成Tensor
        input_ids = torch.tensor(sequence_padding([x.input_ids for x in batch], length=max_len))
        attention_mask = torch.tensor(sequence_padding([x.attention_mask for x in batch], length=max_len))
        # token_type_ids = torch.tensor(sequence_padding([x.token_type_ids for x in batch], length=max_len))
        mask_token_ids = torch.tensor([x.mask_token_id for x in batch])
        labels = torch.tensor([x.label for x in batch]) if not test else None
        batch = InputBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            mask_token_ids=mask_token_ids,
            labels=labels
        )
        return batch


class PromptDataModule(pl.LightningDataModule):

    dataset_mapping = {
        'conll16': CoNll16Dataset,
        'pdtb2': PDTB2Dataset
    }

    def __init__(self, config):
        super().__init__()
        self.config = config.data
        # self.tokenizer = BertTokenizerFast.from_pretrained(config.data.tokenizer)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(config.data.tokenizer)
        print_string("configuration of datamodule")
        print_config(self.config)
        self.DATASET = self.dataset_mapping[self.config.dataset]

    def setup(self, stage=None):
        """
        根据模型运行的阶段，加载对应的数据，并实例化Dataset对象
        """
        if stage == 'fit' or stage is None:
            self.trainset = self.DATASET(self.config, self.tokenizer, dataset_mode='train')
            self.valset = self.DATASET(self.config, self.tokenizer, dataset_mode='val')

        if stage == 'test' or stage is None:
            self.testset = self.DATASET(self.config, self.tokenizer, dataset_mode='test')

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.config.train_bsz, shuffle=True, collate_fn=self.trainset.collate_fn, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.config.val_bsz, shuffle=False, collate_fn=self.valset.collate_fn, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.config.test_bsz, shuffle=False, collate_fn=self.testset.collate_fn, num_workers=4)


def test_batch():
    config = Config.create({
        "data": {
            'dataset': 'pdtb2',
            'tokenizer': "roberta-large",
            'train_path': os.path.join(PROJ_DIR, "data/causal_relation/pdtb2/dataset/train.csv"),
            'val_path': os.path.join(PROJ_DIR, "data/causal_relation/pdtb2/dataset/dev.csv"),
            'test_path': os.path.join(PROJ_DIR, "data/causal_relation/pdtb2/dataset/test.csv"),
            'mapping_path': os.path.join(PROJ_DIR, "data/causal_relation/pdtb2/pdtb_label2word_implicit.json"),
            'train_bsz': 32,
            'val_bsz': 32,
            'test_bsz': 32,
            'cache_dir': './cached/data_utils',
            'force_reload': True
        }
    })
    dm = PromptDataModule(config)
    dm.setup('fit')
    print_string("one sample example")
    print(dm.trainset.samples[0])
    print_string("one feature example")
    print(dm.trainset.features[0])
    print_string("one batch example")
    for batch in dm.train_dataloader():
        batch.info()
        break


if __name__ == '__main__':
    # test_batch()
    lm = LabelMapping(os.path.join(PROJ_DIR, "data/causal_relation/pdtb2/pdtb_label2word_implicit.json"))
    pprint(sorted(lm.words2label.items(), key=lambda x:x[1]))