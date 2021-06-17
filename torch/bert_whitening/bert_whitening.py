'''
Descripttion: 
version: 
Author: Shicript
Date: 2021-06-17 09:58:56
LastEditors: Shicript
LastEditTime: 2021-06-17 11:57:57
'''
import json
import torch
import transformers
import numpy as np
from transformers import BertModel, BertTokenizer
from tokenizers import BertWordPieceTokenizer

from .utils import *


model_path = ''
vocab_file = ''
whiten_file_path = ''
use_fast_tokenizer = True
is_train_whiten = True
CUDA_DEVICE = 'cuda:2'
POOLING = 'cls'

MAX_LENGTH = 128  # 句子长度
N_SIZE = 256  # 降维维度


model = BertModel.from_pretrained(model_path)


# 使用tokenizers加速tokenize
if use_fast_tokenizer:
    tokenizer = BertWordPieceTokenizer(
        vocab='',
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=True,
        lowercase=True
    )
    tokenizer.enable_padding(length=MAX_LENGTH)
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
else:
    tokenizer = BertTokenizer.from_pretrained('')


# 训练白化参数
if not is_train_whiten:
    kernel, bias = load_whiten(whiten_file_path)
    kernel = kernel[:, :N_SIZE]
    print("Loading kernel and bias")


def sents_to_vecs(sents, tokenizer, model, pooling, max_length):
    tokenizer.enable_padding(length=max_length)
    tokenizer.enable_truncation(max_length=max_length)
    with torch.no_grad():
        # inputs = tokenizer.batch_encode_plus(sents, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        bwpt_batch_ids = tokenizer.encode_batch(sents, add_special_tokens=True)
        inputs = transformers.tokenization_utils_base.BatchEncoding()

        inputs['input_ids'] = []
        inputs['token_type_ids'] = []
        inputs['attention_mask'] = []

        for bwpt_batch_id in bwpt_batch_ids:
            inputs['input_ids'].append(bwpt_batch_id.ids)
            inputs['token_type_ids'].append(bwpt_batch_id.type_ids)
            inputs['attention_mask'].append(bwpt_batch_id.attention_mask)

        np_input_list = np.array(inputs['input_ids']).astype(int)
        np_token_type_ids = np.array(inputs['token_type_ids']).astype(int)
        np_attention_mask = np.array(inputs['attention_mask']).astype(int)

        inputs['input_ids'] = torch.from_numpy(np_input_list).to(CUDA_DEVICE)
        inputs['token_type_ids'] = torch.from_numpy(
            np_token_type_ids).to(CUDA_DEVICE)
        inputs['attention_mask'] = torch.from_numpy(
            np_attention_mask).to(CUDA_DEVICE)

        hidden_states = model(**inputs, return_dict=True,
                              output_hidden_states=True).hidden_states

        if pooling == 'first_last_avg':
            output_hidden_state = (
                hidden_states[-1] + hidden_states[1]).mean(dim=1)
        elif pooling == 'last_avg':
            output_hidden_state = (hidden_states[-1]).mean(dim=1)
        elif pooling == 'last2avg':
            output_hidden_state = (
                hidden_states[-1] + hidden_states[-2]).mean(dim=1)
        elif pooling == 'cls':
            output_hidden_state = (hidden_states[-1])[:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(POOLING))
        vec = output_hidden_state.cpu().numpy()

    return vec


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(s ** 0.5))
    W = np.linalg.inv(W.T)
    return W, -mu


def train_white(a_train_sents, b_train_sents, batch_size=1000):
    a_train_batch, b_train_batch = [], []
    a_train_vecs, b_train_vecs = None, None
    batch_num = 0
    for index, sent in enumerate(a_train_sents):
        a_train_batch.append(sent)
        b_train_batch.append(b_train_sents[index])
        if index % 1000 == 0:
            batch_num += 1
            a_batch_vecs = sents_to_vecs(
                a_train_batch, tokenizer, model, pooling=POOLING, max_length=MAX_LENGTH)
            b_batch_vecs = sents_to_vecs(
                b_train_batch, tokenizer, model, pooling=POOLING, max_length=MAX_LENGTH)
            if batch_num == 1:
                a_train_vecs, b_train_vecs = a_batch_vecs, b_batch_vecs
            else:
                a_train_vecs = np.concatenate(
                    (a_train_vecs, a_batch_vecs), axis=0)
                b_train_vecs = np.concatenate(
                    (b_train_vecs, b_batch_vecs), axis=0)
            a_train_batch, b_train_batch = [], []

    a_batch_vecs = sents_to_vecs(
        a_train_batch, tokenizer, model, pooling=POOLING, max_length=MAX_LENGTH)
    b_batch_vecs = sents_to_vecs(
        b_train_batch, tokenizer, model, pooling=POOLING, max_length=MAX_LENGTH)
    a_train_vecs = np.concatenate(
        (a_train_vecs, a_batch_vecs), axis=0)
    b_train_vecs = np.concatenate(
        (b_train_vecs, b_batch_vecs), axis=0)

    print("开始训练白化参数")
    kernel_train, bias_train = compute_kernel_bias(
        [a_train_vecs, b_train_vecs])
    save_whiten(whiten_file_path, kernel_train, bias_train)
    print("保存白化参数成功")


def test_whiten(a_sents: list, b_sents: list, labels: list):
    a_vecs = sents_to_vecs(a_sents, tokenizer, model, pooling=POOLING, max_length=MAX_LENGTH)
    b_vecs = sents_to_vecs(b_sents, tokenizer, model, pooling=POOLING, max_length=MAX_LENGTH)
    a_vecs = transform_and_normalize(a_vecs, kernel, bias)
    b_vecs = transform_and_normalize(b_vecs, kernel, bias)

    sims = (a_vecs, b_vecs).sum(axis=1)
    print(u'Spearmanr corr in Testing set：%s' % calc_spearmanr_corr(labels, sims))


if __name__ == '__main__':
    test_file_path = ''
    a_test_sents, b_test_sents, test_labels = load_dataset(test_file_path)
    test_whiten(a_test_sents, b_test_sents, test_labels)