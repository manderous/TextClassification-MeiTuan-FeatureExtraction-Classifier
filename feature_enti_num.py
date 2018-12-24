import matplotlib.pyplot as plt
import codecs # 写入txt文件

'''
import nltk
tokens = nltk.word_tokenize(train_data)  #分词
tagged = nltk.pos_tag(tokens)  #词性标注
entities = nltk.chunk.ne_chunk(tagged)  #命名实体识别
'''


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
def figure_attribution(pos_label, label_num):
    plt.hist(pos_label, label_num)
    plt.xlabel('pos_label')
    plt.ylabel('pos_label_num')
    # plt.axis([0, 24, 0, 1000]) # 限定坐标的范围
    plt.show()


# 导入训练集的词性文本
pos = load_file('./lib/train_pos.txt')

# 统计每句话中的名词（在这里，我认为名词就是实体）
enti_num = []
for i in range(len(pos)):
    num = 0 # 初始化每句话的名词个数
    for j in range(len(pos[i])):
        spos = pos[i][j]
        if spos == 'n':
            num += 1
        elif spos == 'ng':
            num += 1
        elif spos == 'nl':
            num += 1
        elif spos == 'nr':
            num += 1
        elif spos == 'ns':
            num += 1
        elif spos == 'nt':
            num += 1
        elif spos == 'nz':
            num += 1
    enti_num.append(num)

# 情感词个数的结果写入文件
write_file(enti_num, './lib/train_feature_enti_num.txt')

# 画出特征的直方图分布
figure_attribution(enti_num, max(enti_num))