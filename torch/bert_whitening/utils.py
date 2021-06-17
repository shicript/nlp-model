'''
Descripttion: 
version: 
Author: Shicript
Date: 2021-06-17 11:25:23
LastEditors: Shicript
LastEditTime: 2021-06-17 11:55:46
'''
import json
import pickle
import scipy.stats
import numpy as np

def load_dataset(file_path):
    '''
    加载训练语料,训练语料格式{'src':'原句', 'tgt':'相似句', 'label':int}
    '''
    file = open(file_path, encoding="utf-8")
    sents_a, sents_b, labels = [], [], []
    for line in file.readlines():
        train_data = json.loads(line)
        a_train_data, b_train_data, label = train_data["src"], train_data["tgt"], train_data["label"]
        sents_a.append(a_train_data)
        sents_b.append(b_train_data)
        labels.append(label)

    assert len(sents_a) == len(sents_b)
    return sents_a, sents_b, labels


def load_whiten(path):
    with open(path, 'rb') as f:
        whiten = pickle.load(f)
    kernel = whiten['kernel']
    bias = whiten['bias']
    return kernel, bias


def save_whiten(path, kernel, bias):
    whiten = {
        'kernel': kernel,
        'bias': bias
    }
    with open(path, 'wb') as f:
        pickle.dump(whiten, f)
    return path


def transform_and_normalize(vecs, kernel, bias):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)


def calc_spearmanr_corr(x, y):
    return scipy.stats.spearmanr(x, y).correlation
