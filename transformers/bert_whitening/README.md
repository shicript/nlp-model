<!--
 * @Descripttion: 
 * @version: 
 * @Author: Shicript
 * @Date: 2021-06-17 09:58:40
 * @LastEditors: Shicript
 * @LastEditTime: 2021-06-17 12:00:49
-->
# bert-whitening
通过简单的线性变换（BERT-whitening）操作，效果基本上能媲美BERT-flow模型，这表明往句向量模型里边引入flow模型可能并非那么关键，它对分布的校正可能仅仅是浅层的，而通过线性变换直接校正句向量的协方差矩阵就能达到相近的效果。同时，BERT-whitening还支持降维操作，能达到提速又提效的效果。

原始仓库:

[https://github.com/bojone/BERT-whitening](https://github.com/bojone/BERT-whitening)