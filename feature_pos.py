import matplotlib.pyplot as plt
import codecs # 写入txt文件

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
        for j in range(len(data[i])):
            fin.write(str(data[i][j])+" ")
        fin.write("\n")
    fin.close()

# 构造了7个特征对应的函数
def feature_pos1(pos_row, pos_len, pos_label):
    for i in range(pos_len):
        if pos_row[i] == 'a' or \
        pos_row[i] == 'ad' or \
        pos_row[i] == 'ag' or \
        pos_row[i] == 'al' or \
        pos_row[i] == 'an':
            if i+1 < pos_len:
                if pos_row[i+1] == 'a' or \
                pos_row[i+1] == 'ad' or \
                pos_row[i+1] == 'ag' or \
                pos_row[i+1] == 'al' or \
                pos_row[i+1] == 'an':
                    pos_label[-1].append(1)

def feature_pos2(pos_row, pos_len, pos_label):
    for i in range(pos_len):
        if pos_row[i] == 'a' or \
        pos_row[i] == 'ad' or \
        pos_row[i] == 'ag' or \
        pos_row[i] == 'al' or \
        pos_row[i] == 'an':
            if i+1 < pos_len:
                if pos_row[i+1] == 'n' or \
                pos_row[i+1] == 'ng' or \
                pos_row[i+1] == 'nl' or \
                pos_row[i+1] == 'nr' or \
                pos_row[i+1] == 'ns' or \
                pos_row[i+1] == 'nt' or \
                pos_row[i+1] == 'nz':
                    pos_label[-1].append(2)

def feature_pos3(pos_row, pos_len, pos_label):
    for i in range(pos_len):
        if pos_row[i] == 'a' or \
        pos_row[i] == 'ad' or \
        pos_row[i] == 'ag' or \
        pos_row[i] == 'al' or \
        pos_row[i] == 'an':
            if i+1 < pos_len and pos_row[i+1] == 'uj':
                if i+2 < pos_len:
                    if pos_row[i+2] == 'n' or \
                    pos_row[i+2] == 'ng' or \
                    pos_row[i+2] == 'nl' or \
                    pos_row[i+2] == 'nr' or \
                    pos_row[i+2] == 'ns' or \
                    pos_row[i+2] == 'nt' or \
                    pos_row[i+2] == 'nz':
                        pos_label[-1].append(3)

def feature_pos4(pos_row, pos_len, pos_label):
    for i in range(pos_len):
        if pos_row[i] == 'd' or pos_row[i] == 'zg':
            if i+1 < pos_len:
                if pos_row[i+1] == 'a' or \
                pos_row[i+1] == 'ad' or \
                pos_row[i+1] == 'ag' or \
                pos_row[i+1] == 'al' or \
                pos_row[i+1] == 'an':
                    pos_label[-1].append(4)

# vd vf vg vi vl vn vs vx vy
def feature_pos5(pos_row, pos_len, pos_label):
    for i in range(pos_len):
        if pos_row[i] == 'd' or pos_row[i] == 'zg':
            if i+1 < pos_len:
                if pos_row[i+1] == 'v' or \
                pos_row[i+1] == 'vd' or \
                pos_row[i+1] == 'vf' or \
                pos_row[i+1] == 'vg' or \
                pos_row[i+1] == 'vi' or \
                pos_row[i+1] == 'vl' or \
                pos_row[i+1] == 'vn' or \
                pos_row[i+1] == 'vs' or \
                pos_row[i+1] == 'vx' or \
                pos_row[i+1] == 'vy':
                    pos_label[-1].append(5)

def feature_pos6(pos_row, pos_len, pos_label):
    for i in range(pos_len):
        if pos_row[i] == 'd':
            if i+1 < pos_len:
                if pos_row[i+1] == 'd' or pos_row[i+1] == 'zg':
                    if i+2 < pos_len:
                        if pos_row[i+2] == 'a' or \
                        pos_row[i+2] == 'ad' or \
                        pos_row[i+2] == 'ag' or \
                        pos_row[i+2] == 'al' or \
                        pos_row[i+2] == 'an':
                            pos_label[-1].append(6)

def feature_pos7(pos_row, pos_len, pos_label):
    for i in range(pos_len):
        if pos_row[i] == 'n' or \
        pos_row[i] == 'ng' or \
        pos_row[i] == 'nl' or \
        pos_row[i] == 'nr' or \
        pos_row[i] == 'ns' or \
        pos_row[i] == 'nt' or \
        pos_row[i] == 'nz':
            if i+1 < pos_len:
                if pos_row[i+1] == 'a' or \
                pos_row[i+1] == 'ad' or \
                pos_row[i+1] == 'ag' or \
                pos_row[i+1] == 'al' or \
                pos_row[i+1] == 'an':
                    pos_label[-1].append(7)


#多维list转化为1维度list（相当于把列表压平）,递归
def getnewList(newlist):
	d = []
	for element in newlist:
		if not isinstance(element,list):
			d.append(element)
		else:
			d.extend(getnewList(element))
	return d


# 画出特征的直方图分布
def figure_attribution(pos_label, label_num):
    plt.hist(pos_label, label_num)
    plt.xlabel('pos_label')
    plt.ylabel('pos_label_num')
    # plt.axis([0, 24, 0, 1000]) # 限定坐标的范围
    plt.show()


def call_feature(pos):
    pos_label = [] # 初始化词性组合模式的标签为0
    for i in range(len(pos)):
        pos_label.append([])
        pos_row = pos[i] # 句子的某一行（词性）
        pos_len = len(pos[i]) # 句子的长度（词性）
        feature_pos1(pos_row, pos_len, pos_label)
        feature_pos2(pos_row, pos_len, pos_label)
        feature_pos3(pos_row, pos_len, pos_label)
        feature_pos4(pos_row, pos_len, pos_label)
        feature_pos5(pos_row, pos_len, pos_label)
        feature_pos6(pos_row, pos_len, pos_label)
        feature_pos7(pos_row, pos_len, pos_label)
        if len(pos_label[i]) == 0: # 如果一句话不具有以上任何一种频繁词序列模式，则赋值0
            pos_label[-1].append(0)
    return pos_label


# 导入训练集的词性文本
pos = load_file('./lib/train_pos.txt')
pos_label = call_feature(pos)

# 词性组合模式的结果写入文件
write_file(pos_label, './lib/train_feature_pos.txt')

 # 画出特征的直方图分布
figure_attribution(getnewList(pos_label), 7)
# pos_label是词性组合模式得到的特征！！！