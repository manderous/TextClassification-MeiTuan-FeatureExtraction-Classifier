# TextClassification-MeiTuan-FeatureExtraction-Classifier

这里是利用python3.6进行“特征提取+分类器”来实现美团评论的文本二分类问题。在特征提取部分提取了6种特征，分类器选择了python里面的包xgboost和lightGBM分别实现提升树和GBDT（梯度提升决策树）。现在，最终的结果（准确率和AUC）不是特别好，希望通过调节分类器模型和参数最终可以有一个比较好的结果。


其中：<br>
* （1）-（4）：数据预处理<br>
* （5）-（8）：一层、两层以及双向lstm模型<br>
* （9）-（12）：灵敏度分析<br>
* （13）：lib文件夹下的数据文件介绍<br>

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
