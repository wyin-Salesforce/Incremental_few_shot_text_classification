# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import codecs
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from scipy.stats import beta
from torch.nn import CrossEntropyLoss, MSELoss
from scipy.special import softmax
# from scipy.stats import pearsonr, spearmanr
# from sklearn.metrics import matthews_corrcoef, f1_score

from transformers.tokenization_roberta import RobertaTokenizer
from transformers.optimization import AdamW
from transformers.modeling_roberta import RobertaModel#RobertaForSequenceClassification


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel
# import torch.nn as nn

bert_hidden_dim = 1024
pretrain_model_dir = 'roberta-large' #'roberta-large' , 'roberta-large-mnli', 'bert-large-uncased'

def store_transformers_models(model, tokenizer, output_dir, flag_str):
    '''
    store the model
    '''
    output_dir+='/'+flag_str
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    print('starting model storing....')
    # model.save_pretrained(output_dir)
    torch.save(model.state_dict(), output_dir)
    # tokenizer.save_pretrained(output_dir)
    print('store succeed')

class RobertaForSequenceClassification(nn.Module):
    def __init__(self, tagset_size):
        super(RobertaForSequenceClassification, self).__init__()
        self.tagset_size = tagset_size

        self.roberta_single= RobertaModel.from_pretrained(pretrain_model_dir)
        self.roberta_single.load_state_dict(torch.load('/export/home/Dataset/BERT_pretrained_mine/MNLI_pretrained/_acc_0.9040886899918633.pt'), strict=False)
        self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

    def forward(self, input_ids, input_mask):
        outputs_single = self.roberta_single(input_ids, input_mask, None)
        hidden_states_single = outputs_single[1]#torch.tanh(self.hidden_layer_2(torch.tanh(self.hidden_layer_1(outputs_single[1])))) #(batch, hidden)

        score_single = self.single_hidden2tag(hidden_states_single) #(batch, tag_set)
        return score_single



class RobertaClassificationHead(nn.Module):
    """wenpeng overwrite it so to accept matrix as input"""

    def __init__(self, bert_hidden_dim, num_labels):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

    def forward(self, features):
        x = features#[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, hypo_class=None, premise_class=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.hypo_class = hypo_class
        self.premise_class = premise_class


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, test_class_id, gold_class_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.test_class_id = test_class_id
        self.gold_class_id = gold_class_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def load_Base_train(self, filename):
        '''
        classes: ["entailment", "neutral", "contradiction"]
        '''

        class2example_list={}
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        for row in readfile:
            parts = row.strip().split('\t')
            # print('parts:', parts)
            class_name = parts[0].strip()
            sent_list = class2example_list.get(class_name)
            if sent_list is None:
                sent_list = []
            sent = parts[1].strip()
            sent_list.append(sent)
            class2example_list[class_name] = sent_list
        class_size = len(class2example_list)
        '''now build pos and neg pairs'''
        '''pos pairs'''
        examples_pos = []
        for key, valuelist in class2example_list.items():
            value_size = len(valuelist)
            for i in range(value_size):
                for j in range(value_size):
                    if i != j:
                        # they can be exchanged, but can not be the same
                        examples_pos.append(
                            InputExample(guid='base', text_a=valuelist[i], text_b=valuelist[j], label='pos', hypo_class = key, premise_class = key))

        print('examples_pos size:', len(examples_pos))
        '''neg pairs, we keep the same size of the pos pairs'''
        examples_neg=[]
        for class_i, ex_list in class2example_list.items():
            other_ex_list = []
            for class_j, ex_list_j in class2example_list.items():
                if class_i != class_j:
                    other_ex_list+=ex_list_j
            other_ex_list = random.sample(other_ex_list, len(ex_list))
            for ex_i in ex_list:
                for ex_j in other_ex_list:
                    examples_neg.append(
                        InputExample(guid='base', text_a=ex_i, text_b=ex_j, label='neg',  hypo_class = class_i, premise_class = class_i))


        examples_pos = examples_pos[:200]
        examples_neg = examples_neg[:200]
        print('examples_pos size:', len(examples_pos), ' examples_neg size:', len(examples_neg))
        return examples_pos+ examples_neg, class2example_list


    def load_Base_dev_or_test(self, filename, base_class2example_list, class_2_ood):
        '''
        classes: ["entailment", "neutral", "contradiction"]
        '''

        examples = []
        readfile = codecs.open(filename, 'r', 'utf-8')
        line_co=0
        for row in readfile:
            parts = row.strip().split('\t')
            gold_class = class_2_ood.get(parts[0].strip())
            sent = parts[1].strip()
            '''this sent compares with all base examples'''
            # print('each test example compares with', len(base_class2example_list.keys()), ' base classes')
            for test_class, ex_list in base_class2example_list.items():
                test_class = class_2_ood.get(test_class)
                for ex in ex_list:
                    examples.append(
                        InputExample(guid='base', text_a=sent, text_b=ex, label='neg', hypo_class = test_class, premise_class = gold_class))

        print('examples size:', len(examples))
        return examples



    def get_labels(self):
        'here we keep the three-way in MNLI training '
        return ["entailment", "not_entailment"]
        # return ["entailment", "neutral", "contradiction"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



def convert_examples_to_features(examples, label_list, class_list, max_seq_length,
                                 tokenizer, output_mode,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 sep_token_extra=False,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label : i for i, label in enumerate(label_list)}
    class_map = {label : i for i, label in enumerate(class_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 4 if sep_token_extra else 3
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens_a) > max_seq_length - special_tokens_count:
                tokens_a = tokens_a[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        if cls_token_at_end:
            tokens = tokens + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)


        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              test_class_id = class_map[example.hypo_class],
                              gold_class_id = class_map[example.premise_class]))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()




def load_class_names():
    readfile = codecs.open('/export/home/Dataset/incrementalFewShotTextClassification/category_split.txt', 'r', 'utf-8')
    class_list = set()
    class_2_ood = {}
    for line in readfile:
        class_i = line.strip().split('\t')[1]
        class_2_ood[class_i] = class_i
        if line.strip().split('\t')[0] == 'ood':
            class_2_ood[class_i] = 'ood'
            class_i = 'ood'
        class_list.add(class_i)
    print('class_list size:', len(class_list))
    return list(class_list), class_2_ood


def examples_to_features(source_examples, label_list, class_list,  args, tokenizer, batch_size, output_mode, dataloader_mode='sequential'):
    source_features = convert_examples_to_features(
        source_examples, label_list, class_list, args.max_seq_length, tokenizer, output_mode,
        cls_token_at_end=False,#bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,#2 if args.model_type in ['xlnet'] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=True,#bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=False,#bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0)#4 if args.model_type in ['xlnet'] else 0,)

    dev_all_input_ids = torch.tensor([f.input_ids for f in source_features], dtype=torch.long)
    dev_all_input_mask = torch.tensor([f.input_mask for f in source_features], dtype=torch.long)
    dev_all_segment_ids = torch.tensor([f.segment_ids for f in source_features], dtype=torch.long)
    dev_all_label_ids = torch.tensor([f.label_id for f in source_features], dtype=torch.long)
    dev_all_test_class_ids = torch.tensor([f.test_class_id for f in source_features], dtype=torch.long)
    dev_all_gold_class_ids = torch.tensor([f.gold_class_id for f in source_features], dtype=torch.long)

    dev_data = TensorDataset(dev_all_input_ids, dev_all_input_mask, dev_all_segment_ids, dev_all_label_ids, dev_all_test_class_ids, dev_all_gold_class_ids)
    if dataloader_mode=='sequential':
        dev_sampler = SequentialSampler(dev_data)
    else:
        dev_sampler = RandomSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)


    return dev_dataloader


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")


    args = parser.parse_args()


    processors = {
        "rte": RteProcessor
    }

    output_modes = {
        "rte": "classification"
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")


    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))



    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    class_list, class_2_ood = load_class_names()
    # print('class_list:', class_list)
    base_train_examples, base_class2example_list = processor.load_Base_train('/export/home/Dataset/incrementalFewShotTextClassification/base_train.txt')


    base_truncate_example_size = 0

    base_selected_class2example_list = {}
    for class_i, ex_list in base_class2example_list.items():
        truncate_ex_list = random.sample(ex_list, 1)
        base_selected_class2example_list[class_i] = truncate_ex_list
        base_truncate_example_size+=len(truncate_ex_list)

    # class2example_list = new_class2example_list
    base_dev_examples = processor.load_Base_dev_or_test('/export/home/Dataset/incrementalFewShotTextClassification/base_val.txt', base_selected_class2example_list, class_2_ood)
    base_test_examples = processor.load_Base_dev_or_test('/export/home/Dataset/incrementalFewShotTextClassification/base_test.txt', base_selected_class2example_list, class_2_ood)
    label_list = ["pos", "neg"]

    num_labels = len(label_list)
    print('num_labels:', num_labels, 'base training size:', len(base_train_examples), 'base dev size:', len(base_dev_examples), 'base test size:', len(base_test_examples))

    model = RobertaForSequenceClassification(num_labels)
    tokenizer = RobertaTokenizer.from_pretrained(pretrain_model_dir, do_lower_case=args.do_lower_case)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters,
                             lr=args.learning_rate)
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_test_acc = 0.0
    max_dev_acc = 0.0
    if args.do_train:



        base_train_dataloader = examples_to_features(base_train_examples, label_list, class_list, args, tokenizer, args.train_batch_size, "classification", dataloader_mode='random')
        base_dev_dataloader = examples_to_features(base_dev_examples, label_list, class_list, args, tokenizer, args.train_batch_size, "classification", dataloader_mode='sequential')
        base_test_dataloader = examples_to_features(base_test_examples, label_list, class_list, args, tokenizer, args.train_batch_size, "classification", dataloader_mode='sequential')



        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(base_train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)

        iter_co = 0
        final_test_performance = 0.0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(base_train_dataloader, desc="Iteration")):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, _, _ = batch


                logits = model(input_ids, input_mask)
                loss_fct = CrossEntropyLoss()

                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                iter_co+=1

            '''
            start evaluate on dev set after this epoch
            '''
            model.eval()
            logger.info("***** Running dev *****")
            logger.info("  Num examples = %d", len(base_dev_examples))



            for idd, dev_or_test_dataloader in enumerate([base_dev_dataloader, base_test_dataloader]):


                eval_loss = 0
                nb_eval_steps = 0
                preds = []
                gold_label_ids = []
                all_hypo_class_ids = []
                all_premise_class_ids = []
                # print('Evaluating...')
                for _, batch in enumerate(tqdm(dev_or_test_dataloader, desc="evaluating")):
                    input_ids, input_mask, segment_ids, label_ids, test_class_ids, gold_class_ids = batch
                    # for input_ids, input_mask, segment_ids, label_ids, test_class_ids, gold_class_ids in dev_or_test_dataloader:
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    gold_label_ids+=list(label_ids.detach().cpu().numpy())
                    all_hypo_class_ids+=list(test_class_ids.detach().cpu().numpy())
                    all_premise_class_ids+=list(gold_class_ids.detach().cpu().numpy())


                    with torch.no_grad():
                        logits = model(input_ids, input_mask)

                    '''combine with logits from source domain'''
                    # print('logits:', logits)
                    # print('logits_from_source:', logits_from_source)
                    # weight = 0.9
                    # logits = weight*logits+(1.0-weight)*torch.sigmoid(logits_from_source)
                    if len(preds) == 0:
                        preds.append(logits.detach().cpu().numpy())
                    else:
                        preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

                preds = preds[0]

                pred_probs = softmax(preds,axis=1)
                pos_probs = list(pred_probs[:, 0])
                assert len(pos_probs) == len(all_hypo_class_ids)
                assert len(pos_probs) == len(all_premise_class_ids)
                assert len(pos_probs)%base_truncate_example_size == 0
                test_instance_size = len(pos_probs)//base_truncate_example_size
                if idd ==0:
                    assert test_instance_size ==  len(base_dev_examples)//base_truncate_example_size
                else:
                    assert test_instance_size ==  len(base_test_examples)//base_truncate_example_size

                '''for each test example'''
                # for each_test_id in range(test_instance_size):


                hit_ood_co = 0
                hit_kd_co = 0
                total_ood_co = 0
                total_kd_co =0
                for each_test_id in range(test_instance_size):
                    gold_class_set = set(all_premise_class_ids[each_test_id*base_truncate_example_size: (each_test_id+1)*base_truncate_example_size])
                    assert len(gold_class_set) == 1
                    gold_class = list(gold_class_set)[0] # can be in the base or ood

                    sub_pos_probs = pos_probs[each_test_id*base_truncate_example_size: (each_test_id+1)*base_truncate_example_size]
                    sub_hypo_class_ids = all_hypo_class_ids[each_test_id*base_truncate_example_size: (each_test_id+1)*base_truncate_example_size]

                    hypo_class_2_problist = {}
                    for i in range(base_truncate_example_size):
                        hypo_class_i = sub_hypo_class_ids[i]
                        problist = hypo_class_2_problist.get(hypo_class_i)
                        if problist is None:
                            problist = []
                        problist.append(sub_pos_probs[i])
                        hypo_class_2_problist[hypo_class_i] = problist

                    '''find the base class with max prob'''
                    final_pred_class = -1
                    max_prob = 0.0
                    for hypo_class_i, problist  in hypo_class_2_problist.items():
                        mean_prob = np.mean(problist)
                        if mean_prob > max_prob:
                            max_prob = mean_prob
                            final_pred_class = hypo_class_i

                    if max_prob < 0.5:
                        #00d
                        final_pred_class = class_list.index('ood')


                    '''compute acc'''
                    if gold_class == class_list.index('ood'):
                        total_ood_co+=1
                    else:
                        total_kd_co+=1

                    if final_pred_class == gold_class:
                        if gold_class == class_list.index('ood'):
                            hit_ood_co +=1
                        else:
                            hit_kd_co+=1

                acc_ood = hit_ood_co/(1e-8+total_ood_co)
                acc_kd = hit_kd_co/(1e-8+total_kd_co)
                if idd == 0:
                    print(' dev: acc_kd:', acc_kd, 'acc_ood:', acc_ood)
                else:
                    print(' test: acc_kd:', acc_kd, 'acc_ood:', acc_ood)


if __name__ == "__main__":
    main()

'''

CUDA_VISIBLE_DEVICES=2 python -u train.kazuma.py --task_name rte --do_train --do_lower_case --num_train_epochs 20 --train_batch_size 32 --eval_batch_size 64 --learning_rate 1e-6 --max_seq_length 128 --seed 42


'''
