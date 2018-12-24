import numpy as np
import jieba # 分词
import jieba.posseg as psg # 词性标注
import codecs # 写入txt文件

# jieba.load_userdict("百度分词词库.txt") # 可以扩展jieba分词的词库

'''
# 测试数据
import linecache # 读取文件中的某一行
file_name = 'train.txt'
linecache.getline(file_name, 1) # 读取文件中的某一行
a = linecache.getline(file_name, 1).strip().split()
a[0] = int(a[0])
b = [[x.word,x.flag] for x in psg.cut(a[1])]
c = [x for x in jieba.cut(a[1])]
'''

# 读取txt文件，并且转成数组类型
def load_file(file_name):
    data = [] # 保存数据
    with open(file_name, encoding='utf-8') as fin:
        for line in fin:
            line = line.strip().split()
            data.append(line)
    return np.array(data)

# 分词与词性标注
def cut_pos_file(data):
    data_cut = [] # 保存分词结果
    data_w_pos = [] # 保存分词与词性标注结果
    data_pos = [] # 仅保存词性标注结果
    for i in range(len(data)):
        data_cut.append([])
        data_w_pos.append([])
        data_pos.append([])
        [data_cut[-1].append(x) for x in jieba.cut(data[i])]
        for x in psg.cut(data[i]):
            data_pos[-1].append(x.flag)
            data_w_pos[-1].append([x.word,x.flag])
    return data_cut, data_pos, data_w_pos

# 写入txt文件
def write_file(data, file_name):
    fin = codecs.open(file_name, "w", "UTF-8")
    for i in range(len(data)):
        for j in range(len(data[i])):
            fin.write(str(data[i][j])+" ")
        fin.write("\n")
    fin.close()


train = load_file('./lib/train.txt')
train_data = train[:,1]
train_label = np.array([int(x) for x in train[:,0]]) # 将字符串数组转换成整型数组
train_data_positive = train_data[train_label == 1] # 提取标签等于1的训练数据
train_data_negative = train_data[train_label ==0] # 提取标签等于0的训练数据
train_cut, train_pos, train_Wpos = cut_pos_file(train_data)
train_positive_cut, train_positive_pos, train_positive_Wpos = cut_pos_file(train_data_positive)
train_negative_cut, train_negative_pos, train_negative_Wpos = cut_pos_file(train_data_negative)

write_file(train_cut, './lib/train_cut.txt')
write_file(train_pos, './lib/train_pos.txt')
write_file(train_Wpos, './lib/train_Wpos.txt')

write_file(train_positive_cut, './lib/train_positive_cut.txt')
write_file(train_positive_pos, './lib/train_positive_pos.txt')
write_file(train_positive_Wpos, './lib/train_positive_Wpos.txt')

write_file(train_negative_cut, './lib/train_negative_cut.txt')
write_file(train_negative_pos, './lib/train_negative_pos.txt')
write_file(train_negative_Wpos, './lib/train_negative_Wpos.txt')



test = load_file('./lib/test.txt')
test_data = test[:,1]
test_label = np.array([int(x) for x in test[:,0]]) # 将字符串数组转换成整型数组
# 提取标签等于1的测试数据，主要用于频繁词序列挖掘，计算词序列的正负类区分度
test_data_positive = test_data[test_label == 1]
test_data_negative = test_data[test_label ==0] # 提取标签等于0的测试数据
test_cut, test_pos, test_Wpos = cut_pos_file(test_data)
test_positive_cut, test_positive_pos, test_positive_Wpos = cut_pos_file(test_data_positive)
test_negative_cut, test_negative_pos, test_negative_Wpos = cut_pos_file(test_data_negative)
