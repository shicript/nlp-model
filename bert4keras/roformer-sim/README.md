<!--
 * @Descripttion: 
 * @version: 
 * @Author: Shicript
 * @Date: 2021-06-16 15:06:57
 * @LastEditors: Shicript
 * @LastEditTime: 2021-07-19 16:10:55
-->
# RoFormer-Sim
追一科技发布的simbert升级版本，以RoFormer为基础模型，对SimBERT整合优化得到升级版本的RoFormer-Sim。

在simbert的基础上相当于将基础架构换成了RoFormer,在训练细节上也有区别。并且升级版的RoFormer-Sim有更多的训练语料，不再仅仅包含百度知道中的疑问句，还扩展到了一般句式，适用场景更大。

模型权重下载及原始仓库地址:
[https://github.com/ZhuiyiTechnology/roformer-sim](https://github.com/ZhuiyiTechnology/roformer-sim)


### 07-19更新
更新追一科技有监督rofromer-sim

```python
>>> compute_similarity("红色的苹果", ["苹果","绿色的苹果", "红苹果", "苹果手机"])
[0.78607535, 0.6974648 , 0.9816927 , 0.7395245 ]
>>> compute_similarity("今天天气不错", ["今天天气很好", "今天天气不好"])
[0.9769764, 0.6236062]
```

经过开源语料有监督的训练之后, 模型在语义相反或者不一致的条件下检索的效果得到一定的提升。