#本脚本主要实现了基于python通过已有的情感词典对文本数据做的情感分析的项目目的
import numpy as np
import matplotlib.pyplot as plt
import codecs # 写入txt文件
 
# 打开情感词典文件，返回列表
def open_dict(Dict, path = './sentiment_dic/'):
    path = path + '%s.txt' %Dict
    dictionary = open(path, 'r', encoding='utf-8',errors='ignore')
    dict = []
    for word in dictionary:
        word = word.strip('\n').strip()
        dict.append(word)
    return dict

# 读取txt文件，这里是为了读取分词后的文本
def load_file(file_name):
    data = [] # 保存数据
    with open(file_name, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip().split()
            data.append(line)
    return data

# 写入txt文件
def write_file(data, file_name):
    fin = codecs.open(file_name, "w", "UTF-8")
    for i in range(len(data)):
        fin.write(str(data[i])+"\n")
    fin.close()

def judgeodd(num):  #往情感词前查找否定词，找完全部否定词，若数量为奇数，乘以-1，若数量为偶数，乘以1.
    if num % 2 == 0:
        return 'even'
    else:
        return 'odd'


def sentiment_score_list(cut):
    count1 = []
    count2 = []
    for segtmp in cut:
        #print(sen)# 循环遍历每一个评论
        #print(segtmp)
        i = 0 #记录扫描到的词的位置
        a = 0 #记录情感词的位置
        poscount = 0 # 积极词的第一次分值
        poscount2 = 0 # 积极反转后的分值
        poscount3 = 0 # 积极词的最后分值（包括叹号的分值）
        negcount = 0
        negcount2 = 0
        negcount3 = 0
        for word in segtmp:
            if word in posdict: # 判断词语是否是积极情感词
                poscount +=1
                c = 0
                for w in segtmp[a:i]: # 扫描情感词前的程度词
                    if w in mostdict:
                        poscount *= 4.0
                    elif w in verydict:
                        poscount *= 3.0
                    elif w in moredict:
                       poscount *= 2.0
                    elif w in ishdict:
                        poscount *= 0.5
                    elif w in insuffhdict:
                        poscount *= 0.3
                    elif w in deny_word: c+= 1
                if judgeodd(c) == 'odd': # 扫描情感词前的否定词数
                    poscount *= -1.0
                    poscount2 += poscount
                    poscount = 0
                    poscount3 = poscount + poscount2 + poscount3
                    poscount2 = 0
                else:
                    poscount3 = poscount + poscount2 + poscount3
                    poscount = 0
                a = i+1
            elif word in negdict: # 消极情感的分析，与上面一致
                negcount += 1
                d = 0
                for w in segtmp[a:i]:
                    if w in mostdict:
                        negcount *= 4.0
                    elif w in verydict:
                        negcount *= 3.0
                    elif w in moredict:
                        negcount *= 2.0
                    elif w in ishdict:
                        negcount *= 0.5
                    elif w in insuffhdict:
                        poscount *= 0.3
                    elif w in degree_word:
                        d += 1
                if judgeodd(d) == 'odd':
                    negcount *= -1.0
                    negcount2 += negcount
                    negcount = 0
                    negcount3 = negcount + negcount2 + negcount3
                    negcount2 = 0
                else:
                    negcount3 = negcount + negcount2 + negcount3
                    negcount = 0
                a = i + 1
            elif word == '！' or word == '!': # 判断句子是否有感叹号
                for w2 in segtmp[::-1]: # 扫描感叹号前的情感词，发现后权值+2，然后退出循环
                    if w2 in posdict:
                        poscount3 += 2
                    elif w2 in negdict:
                        negcount3 += 2
                    else:
                        poscount3 +=0
                        negcount3 +=0
                        break
            else:
                poscount3=0
                negcount3=0
            i += 1
 
            # 以下是防止出现负数的情况
            pos_count = 0
            neg_count = 0
            if poscount3 <0 and negcount3 > 0:
                neg_count = negcount3 - poscount3
                pos_count = 0
            elif negcount3 <0 and poscount3 > 0:
                pos_count = poscount3 - negcount3
                neg_count = 0
            elif poscount3 <0 and negcount3 < 0:
                neg_count = -pos_count
                pos_count = -neg_count
            else:
                pos_count = poscount3
                neg_count = negcount3
            count1.append([pos_count,neg_count]) #返回每条评论打分后的列表
            #print(count1)
        count2.append(count1)
        count1=[]
        #print(count2)
    return count2  #返回所有评论打分后的列表
 
def sentiment_score(senti_score_list):#分析完所有评论后，正式对每句评论打情感分
    score = []
    sentiment_label = []
    for review in senti_score_list:#senti_score_list
        score_array =  np.array(review)
        Pos = np.sum(score_array[:,0])#积极总分
        Neg = np.sum(score_array[:,1])#消极总分
        AvgPos = np.mean(score_array[:,0])#积极情感均值
        AvgPos = float('%.lf' % AvgPos)
        AvgNeg = np.mean(score_array[:, 1])#消极情感均值
        AvgNeg = float('%.1f' % AvgNeg)
        StdPos = np.std(score_array[:, 0])#积极情感方差
        StdPos = float('%.1f' % StdPos)
        StdNeg = np.std(score_array[:, 1])#消极情感方差
        StdNeg = float('%.1f' % StdNeg)
        res=Pos-Neg # 最终情感得分
        # 情感得分细节：总得分，积极得分，消极得分，积极情感均值，消极情感均值，积极情感方差，消极情感方差
        score.append([res, Pos, Neg, AvgPos, AvgNeg, StdPos, StdNeg])
        if res>0:
            sentiment_label.append(2) # 好评
        elif res<0:
            sentiment_label.append(1) # 差评
        else:
            sentiment_label.append(0) # 中评
    return sentiment_label

# 画出特征的直方图分布
def figure_attribution(sentiment_label, label_num):
    plt.hist(sentiment_label, label_num)
    plt.xlabel('sentiment_label')
    plt.ylabel('sentiment_label_num')
    # plt.axis([0, 24, 0, 1000]) # 限定坐标的范围
    plt.show()
 
deny_word = open_dict(Dict='deny')#否定词词典
posdict = open_dict(Dict='positive_sentiment') + open_dict(Dict='positive_comment') #积极词典
negdict = open_dict(Dict = 'negative_sentiment') + open_dict(Dict='negative_comment') # 消极词典
degree_word = open_dict(Dict = 'degree')#程度词词典
 
#为程度词设置权重
mostdict = degree_word[degree_word.index('﻿extreme')+1: degree_word.index('very')] #权重4，即在情感前乘以3
verydict = degree_word[degree_word.index('very')+1: degree_word.index('more')] #权重3
moredict = degree_word[degree_word.index('more')+1: degree_word.index('ish')]#权重2
ishdict = degree_word[degree_word.index('ish')+1: degree_word.index('insufficiently')]#权重0.5
insuffhdict = degree_word[degree_word.index('insufficiently')+1: degree_word.index('over')]#权重0.3
overdict = degree_word[degree_word.index('over')+1: degree_word.index('last')]#权重-0.5，这个程度词先不管


#读取要做情感分析的文本，这里读取的是已经分词后的文本
cut = load_file('./lib/train_cut.txt')

# 返回每句话的情感标签
sentiment_label = sentiment_score(sentiment_score_list(cut))

# 情感模式的结果写入文件
write_file(sentiment_label, './lib/train_feature_sentiment.txt')

# 画出特征的直方图分布
figure_attribution(sentiment_label, 3)