# TextClassification-MeiTuan-FeatureExtraction-Classifier

这里是利用python3.6进行“特征提取+分类器”来实现美团评论的文本二分类问题。在特征提取部分提取了6种特征，分类器选择了python里面的包xgboost和lightGBM分别实现提升树和GBDT（梯度提升决策树）。现在，最终的结果（准确率和AUC）不是特别好，希望通过调节分类器模型和参数最终可以有一个比较好的结果。


其中：<br>
* （1）-（5）：数据预处理<br>
* （6）-（12）：一层、两层以及双向lstm模型<br>
* （13）：分类器<br>
* （14）：feature文件夹下的数据文件介绍<br>
* （15）：lib文件夹下的数据文件介绍<br>
* （16）：model文件夹下的数据文件介绍<br>
* （17）：sentiment_dic文件夹下的数据文件介绍<br>
* （18）：tool文件夹下的数据文件介绍<br>

****

|Author|manderous|
|---|---|
|E-mail|manderous@foxmail.com|

****

## 目录
* [数据预处理](#数据预处理)
    * (1)pretreatment.py
    * (2)bi_tri_gram.py
    * (3)word2vec_test.py
    * (4)zhwiki_2017_03.sg_50d.word2vec
    * (5)PrefixSpan.py
* [特征提取](#特征提取)
    * (6)feature_pos.py
    * (7)feature_fresq.py
    * (8)feature_sentiment.py
    * (9)feature_slen.py
    * (10)feature_senti_num.py
    * (11)feature_enti_num.py
    * (12)import_csv.py
* [分类器](#分类器)
    * (13)boost_tree.py
* [(14)feature文件夹下的数据文件介绍](#feature文件夹下的数据文件介绍)
* [(15)lib文件夹下的数据文件介绍](#lib文件夹下的数据文件介绍)
* [(16)model文件夹下的数据文件介绍](#model文件夹下的数据文件介绍)
* [(17)sentiment_dic文件夹下的数据文件介绍](#sentiment_dic文件夹下的数据文件介绍)
* [(18)tool文件夹下的数据文件介绍](#tool文件夹下的数据文件介绍)

****

## 数据预处理
### pretreatment.py
#### 分词、词性标注、正负文本的划分、训练集测试集的划分（python文件）
输入：<br>
```
./lib/train.txt：训练集<br>
./lib/test.txt：测试集<br>
```
输出：<br>
```
./lib/train_positive_cut.txt：训练集，正标签，的分词结果<br>
./lib/train_positive_pos.txt：训练集，正标签，的词性标注结果<br>
./lib/train_positive_Wpos.txt：训练集，正标签，的分词和词性标注结果<br>
<br>
./lib/train_negative_cut.txt：训练集，负标签，的分词结果<br>
./lib/train_negative_pos.txt：训练集，负标签，的词性标注结果<br>
./lib/train_negative_Wpos.txt：训练集，负标签，的分词和词性标注结果<br>
<br>
./lib/train_cut.txt：训练集，的分词结果<br>
./lib/train_pos.txt'：训练集，的词性标注结果<br>
./lib/train_Wpos.txt：训练集，的分词和词性标注结果<br>
```

### (2)bi_tri_gram.py
#### 生成bi-gram、tri-gram（python文件）
输入：<br>
```
./lib/train_positive_cut.txt：训练集，正标签，的分词结果<br>
./lib/train_negative_cut.txt：训练集，负标签，的分词结果<br>
```
输出：<br>
```
./lib/train_positive_trigram.txt：训练集，正标签，的tri-gram结果<br>
./lib/train_negative_trigram.txt：训练集，负标签，的tri-gram结果<br>
```

### (3)word2vec_test.py
#### 导入搜狗词向量语料加载，将积极文本和消极文本的词语生成词向量，保存至文本文件中（python文件）
输入：<br>
```
./lib/train_positive_cut.txt：训练集，正标签，的分词结果<br>
./lib/train_negative_cut.txt：训练集，负标签，的分词结果<br>
```
输出：<br>
```
./lib/wordsList.npy：数字索引词语变量<br>
./lib/wordIndexVector.npy：数字索引词向量变量<br>
```

### (4)./tool/zhwiki_2017_03.sg_50d.word2vec
#### 搜狗词向量语料（word2vec文件）
百度云资源：https://pan.baidu.com/s/1C94HXCCWOmX-W4IbajXFyA

### (5)PrefixSpan.py
#### 导入搜狗词向量语料加载，将积极文本和消极文本的词语生成词向量，保存至文本文件中（python文件）
输入：<br>
```
./lib/train_positive_trigram.txt：训练集，正标签，的tri-gram结果<br>
./lib/train_negative_trigram.txt：训练集，负标签，的tri-gram结果<br>
```
输出：<br>
```
./lib/train_positive_prefixPattern.txt：训练集，正标签，的所以频繁项（支持度50）<br>
./lib/train_positive_prefixFrequentSub_sup50.txt：训练集，正标签，的所以频繁项（做了一些处理，比如删除中括号，只保留在train_positive_prefixPattern.txt中第一个中括号里面的频繁词序列）（支持度50）<br>
./lib/train_positive_prefixFrequentSub_sup25.txt：训练集，正标签，的所以频繁项（做了一些处理，比如删除中括号，只保留在train_positive_prefixPattern.txt中第一个中括号里面的频繁词序列）（支持度25）<br>
<br>
./lib/train_negative_prefixPattern.txt：训练集，负标签，的所以频繁项（支持度100）<br>
./lib/train_negative_prefixFrequentSub_sup100.txt：训练集，负标签，的所以频繁项（做了一些处理，比如删除中括号，只保留在train_positive_prefixPattern.txt中第一个中括号里面的频繁词序列）（支持度100）<br>
./lib/train_negative_prefixFrequentSub_sup50.txt：训练集，负标签，的所以频繁项（做了一些处理，比如删除中括号，只保留在train_positive_prefixPattern.txt中第一个中括号里面的频繁词序列）（支持度50）<br>
<br>
./lib/train_squence_dict.txt：最终得到的训练集的频繁词序列！！！以及对应的区分度dist（这里最终选出的频繁词序列要求支持度dist>=0.85）<br>
```

****



