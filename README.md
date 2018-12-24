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

## 特征提取
### (6)feature_pos.py
#### 词性组合模式的特征（python文件）
jieba词性对照表：
参考网址：
* 1 jieba分词中所有词性对应字母（词性列表及符号表示）
https://blog.csdn.net/a2099948768/article/details/82216906
* 2 jieba中文分词词性/解释对照表
http://www.niumou.com.cn/183
* 3 重要词性对照：
n/n开头的：名词 ng nl nr ns nt nz<br>
v/v开头的：动词 vd vf vg vi vl vn vs vx vy<br>
a/a开头的：形容词 ad ag al an<br>
zg（部分是副词）/d：副词<br>
uj：助词“的”<br>
* 4 根据佳峰师兄的论文中的词性组合
a + a：1<br>
a + n：2<br>
a + uj + n：3<br>
d/zg + a：4<br>
d/zg + v：5<br>
d + d/zg + a：6<br>
n + a：7<br>
其它：0<br>
输入：<br>
```
./lib/train_pos.txt：训练集，的词性标注结果<br>
```
输出：<br>
```
./lib/train_feature_pos.txt：训练集，词性组合模式的特征<br>
```

### (7)feature_fresq.py
#### 频繁词序列模式的特征（python文件）
根据PrefixSpan.py文件提取出来的，输出字典变量：squence_support_dist_dict_filter，或者输出文件：train_data_squence_dict.txt<br>
一如既往 的 好：1<br>
不错：2<br>
不错 的：3<br>
古色古香：4<br>
味道 不错：5<br>
很 不错：6<br>
很 好：7<br>
很 新鲜：8<br>
得 恰到好处：9<br>
恰到好处：10<br>
新鲜：11<br>
最：12<br>
服务：13<br>
服务态度：14<br>
烤：15<br>
的 恰到好处：16<br>
还：17<br>
还不错：18<br>
其它：0<br>
输入：<br>
```
./lib/train_cut.txt：训练集，的分词结果<br>
```
输出：<br>
```
./lib/train_feature_fresq.txt：训练集，频繁词序列模式的特征<br>
```

### (8)feature_sentiment.py
#### 情感的特征（基于情感词典的情感分类）（python文件）
情感得分<br>
积极：2<br>
消极：1<br>
中性：0<br>
输入：<br>
```
./lib/train_cut.txt：训练集，的分词结果<br>
./sentiment_dic/deny.txt：否定词典<br>
./sentiment_dic/positive_sentiment.txt：正面情感词语<br>
./sentiment_dic/positive_comment.txt：正面评价词语<br>
./sentiment_dic/negative_sentiment.txt：负面情感词语<br>
./sentiment_dic/negative_comment.txt：负面评价词语<br>
./sentiment_dic/degree.txt：程度级别词语（有extreme、very、more、ish、insufficiently、over六个程度等级）<br>
```
输出：<br>
```
./lib/train_feature_sentiment.txt：训练集，情感模式的特征<br>
```

### (9)feature_slen.py
#### 句子长度的特征（python文件）
输入：<br>
```
./lib/train_cut.txt：训练集，的分词结果<br>
```
输出：<br>
```
./lib/train_feature_slen.txt：训练集，句子长度的特征<br>
```

### (10)feature_senti_num.py
#### 情感词的个数（python文件）
输入：<br>
```
./lib/train_cut.txt：训练集，的分词结果<br>
./sentiment_dic/positive_sentiment.txt：正面情感词语<br>
./sentiment_dic/positive_comment.txt：正面评价词语<br>
./sentiment_dic/negative_sentiment.txt：负面情感词语<br>
./sentiment_dic/negative_comment.txt：负面评价词语<br>
```
输出：<br>
```
./lib/train_feature_senti_num.txt：训练集，情感次个数的特征<br>
```

### (11)feature_enti_num.py
#### 实体的个数（python文件）
在这里，我认为所有的名词都是实体，也就是在jieba词性标注中，被标注为n, ng, nl, nr, ns, nt, nz的词语。
输入：<br>
```
./lib/train_pos.txt'：训练集，的词性标注结果<br>
```
输出：<br>
```
./lib/train_feature_enti_num.txt：训练集，情感次个数的特征<br>
```

### (12)import_csv.py
#### 将文本所有的特征导出在一个csv文件中（python文件）
输入：<br>
```
./lib/train.txt：训练集<br>
./lib/train_feature_pos.txt：训练集，词性组合模式的特征<br>
./lib/train_feature_fresq.txt：训练集，频繁词序列模式的特征<br>
./lib/train_feature_sentiment.txt：训练集，情感模式的特征<br>
./lib/train_feature_slen.txt：训练集，句子长度的特征<br>
```
输出：<br>
```
./feature/train.csv：训练集，所有的特征（特征空间）<br>
```

****

## 分类器
### (13)boost_tree.py
#### 利用XGBoost和LightGBM对评论进行分类（python文件）
最后还会显示出XGBoost和LightGBM的损失值。
输入：<br>
```
./feature/train.csv：训练集，所有的特征（特征空间）<br>
```
输出：<br>
```
./feature/train.svm：训练集（原训练集的80%），表示成稀疏矩阵的形式<br>
./feature/valid.svm：验证集（原训练集的20%），表示成稀疏矩阵的形式<br>
```

****

