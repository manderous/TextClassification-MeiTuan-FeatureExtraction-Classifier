import codecs # 写入txt文件
import matplotlib.pyplot as plt

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


# 导入训练集的分词文本
cut = load_file('./lib/train_cut.txt')

# 统计每句话的长度
sentence_length = []
for i in range(len(cut)):
    sentence_length.append(len(cut[i]))

# 句子长度的结果写入文件
write_file(sentence_length, './lib/train_feature_slen.txt')

# 画出特征的直方图分布
figure_attribution(sentence_length, max(sentence_length))