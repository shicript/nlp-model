'''
Descripttion: 
version: 
Author: Shicript
Date: 2021-06-16 15:02:15
LastEditors: Shicript
LastEditTime: 2021-07-19 15:51:01
'''
import numpy as np
from beeprint import pp
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
from bert4keras.snippets import uniout


maxlen = 64

config_path = '/Users/shicript/Downloads/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/Users/shicript/Downloads/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/Users/shicript/Downloads/chinese_roformer-sim-char-ft_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)

roformer = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    application='unilm',
    with_pool='linear'
)

encoder = keras.models.Model(roformer.inputs, roformer.outputs[0])
seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])

'''
roformer-sim生成相似句示例
'''
class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate(
            [segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(seq2seq).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95, mask_idxs=[]):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        for i in mask_idxs:
            token_ids[i] = tokenizer._token_mask_id
        output_ids = self.random_sample([token_ids, segment_ids], n,
                                        topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def gen_synonyms(text, n=100, k=20, mask_idxs=[]):
    ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    '''
    r = synonyms_generator.generate(text, n, mask_idxs=mask_idxs)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]

# seq2seq生成
gen_synonyms("武汉的天气阴晴不定")


'''
encoder转化向量，计算相似度示例
'''


def compute_similarity(a_sent: str, b_sents: list):
    batch_sents = [a_sent] + b_sents
    batch_token_ids, batch_segment_ids = [], []
    for sent in batch_sents:
        token_ids, segment_ids = tokenizer.encode(sent)
        batch_token_ids.append(token_ids)
        batch_segment_ids.append(segment_ids)
    batch_token_ids = sequence_padding(batch_token_ids)
    batch_segment_ids = sequence_padding(batch_segment_ids)
    output = encoder.predict([batch_token_ids, batch_segment_ids])
    pp(output)
    
    output /= (output**2).sum(axis=1, keepdims=True)**0.5
    sim = np.dot(output[1:], output[0])
    return sim


sims = compute_similarity("武汉的天气阴晴不定", ["武汉天气阴晴不定，最近哪几天去武汉比较适合"])
pp(sims)  # 结果 array([0.86337876], dtype=float32)
