import numpy as np
import pandas as pd


# 读取txt文件，并且转成数组类型
# 如果flag是1，会把读取文件的每一行都按空格分割开；如果flag是2，就不分割
def load_file(file_name, flag):
    data = [] # 保存数据
    with open(file_name, encoding='utf-8') as fin:
        for line in fin:
            if flag == 1:
                line = line.strip().split()
            if flag == 2:
                line = line.strip()
            data.append(line)
    return data
    # return np.array(data)


train = np.array(load_file('./lib/train.txt', 1)) # 因为读取train.txt文件需要把句子和标签分开，所以flag=1
feature_pos = load_file('./lib/train_feature_pos.txt', 2) # 词性组合模式
feature_fresq = load_file('./lib/train_feature_fresq.txt', 2) # 频繁词序列模式
feature_sentiment = load_file('./lib/train_feature_sentiment.txt', 2) # 情感打分
feature_slen = load_file('./lib/train_feature_slen.txt', 2) # 句子长度
feature_senti_num = load_file('./lib/train_feature_senti_num.txt', 2) # 情感词个数
feature_enti_num = load_file('./lib/train_feature_enti_num.txt', 2) # 实体个数


#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'Label':train[:,0], 'Sentence':train[:,1], \
                          'feature_pos':feature_pos, 'feature_fresq':feature_fresq, \
                          'feature_sentiment':feature_sentiment, 'feature_slen':feature_slen, \
                          'feature_senti_num':feature_senti_num, 'feature_enti_num':feature_enti_num})

#将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("./feature/train.csv", index=True)


