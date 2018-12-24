import codecs # 写入txt文件
import matplotlib.pyplot as plt

# 打开情感词典文件，返回列表
def open_dict(Dict, path = './sentiment_dic/'):
    path = path + '%s.txt' %Dict
    dictionary = open(path, 'r', encoding='utf-8',errors='ignore')
    dict = []
    for word in dictionary:
        word = word.strip('\n').strip()
        dict.append(word)
    return dict

# 读取txt文件
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

# 画出特征的直方图分布
def figure_attribution(label, label_num):
    plt.hist(label, label_num)
    plt.xlabel('label')
    plt.ylabel('label_num')
    # plt.axis([0, 24, 0, 1000]) # 限定坐标的范围
    plt.show()

# 统计词频
def wordcount(cut):
    count_dict = {}
    for i in range(len(cut)):
        for j in range(len(cut[i])):
            word = cut[i][j]
            if word in count_dict.keys():
                count_dict[word] += 1
            else:
                count_dict[word] = 1
    #按照词频从高到低排列
    count_list=sorted(count_dict.items(),key=lambda x:x[1],reverse=True)
    return count_list
    # return count_dict


# 导入训练集的分词文本
cut = load_file('./lib/train_cut.txt')

# 统计训练集文本的词频
count_list = wordcount(cut)


# 情感词个数的结果写入文件
# write_file(senti_num, './lib/train_feature_senti_num.txt')

# 画出特征的直方图分布
# figure_attribution(senti_num, max(senti_num))
