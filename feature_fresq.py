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
def feature_pos1(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '一如既往':
            if i+1 < cut_len and cut_row[i+1] == '的':
                if i+2 < cut_len and cut_row[i+2] == '好':
                    cut_label[-1].append(1)


def feature_pos2(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '不错':
            cut_label[-1].append(2)

def feature_pos3(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '不错':
            if i+1 < cut_len and cut_row[i+1] == '的':
                    cut_label[-1].append(3)

def feature_pos4(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '古色古香':
            cut_label[-1].append(4)

def feature_pos5(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '味道':
            if i+1 < cut_len and cut_row[i+1] == '不错':
                    cut_label[-1].append(5)

def feature_pos6(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '很':
            if i+1 < cut_len and cut_row[i+1] == '不错':
                    cut_label[-1].append(6)

def feature_pos7(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '很':
            if i+1 < cut_len and cut_row[i+1] == '好':
                    cut_label[-1].append(7)

def feature_pos8(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '很':
            if i+1 < cut_len and cut_row[i+1] == '新鲜':
                    cut_label[-1].append(8)

def feature_pos9(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '得':
            if i+1 < cut_len and cut_row[i+1] == '恰到好处':
                    cut_label[-1].append(9)

def feature_pos10(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '恰到好处':
            cut_label[-1].append(10)

def feature_pos11(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '新鲜':
            cut_label[-1].append(11)

def feature_pos12(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '最':
            cut_label[-1].append(12)

def feature_pos13(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '服务':
            cut_label[-1].append(13)

def feature_pos14(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '服务态度':
            cut_label[-1].append(14)

def feature_pos15(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '烤':
            cut_label[-1].append(15)

def feature_pos16(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '的':
            if i+1 < cut_len and cut_row[i+1] == '恰到好处':
                    cut_label[-1].append(16)

def feature_pos17(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '还':
            cut_label[-1].append(17)

def feature_pos18(cut_row, cut_len, cut_label):
    for i in range(cut_len):
        if cut_row[i] == '还不错':
            cut_label[-1].append(18)

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
def figure_attribution(fresq_label, label_num):
    plt.hist(fresq_label, label_num)
    plt.xlabel('fresq_label')
    plt.ylabel('fresq_label_num')
    # plt.axis([0, 24, 0, 1000]) # 限定坐标的范围
    plt.show()

def call_feature(cut):
    cut_label = [] # 初始化词性组合模式的标签为0
    for i in range(len(cut)):
        cut_label.append([])
        cut_row = cut[i] # 句子的某一行（词性）
        cut_len = len(cut[i]) # 句子的长度（词性）
        feature_pos1(cut_row, cut_len, cut_label)
        feature_pos2(cut_row, cut_len, cut_label)
        feature_pos3(cut_row, cut_len, cut_label)
        feature_pos4(cut_row, cut_len, cut_label)
        feature_pos5(cut_row, cut_len, cut_label)
        feature_pos6(cut_row, cut_len, cut_label)
        feature_pos7(cut_row, cut_len, cut_label)
        feature_pos8(cut_row, cut_len, cut_label)
        feature_pos9(cut_row, cut_len, cut_label)
        feature_pos10(cut_row, cut_len, cut_label)
        feature_pos11(cut_row, cut_len, cut_label)
        feature_pos12(cut_row, cut_len, cut_label)
        feature_pos13(cut_row, cut_len, cut_label)
        feature_pos14(cut_row, cut_len, cut_label)
        feature_pos15(cut_row, cut_len, cut_label)
        feature_pos16(cut_row, cut_len, cut_label)
        feature_pos17(cut_row, cut_len, cut_label)
        feature_pos18(cut_row, cut_len, cut_label)
        if len(cut_label[i]) == 0: # 如果一句话不具有以上任何一种频繁词序列模式，则赋值0
            cut_label[-1].append(0)
    return cut_label


cut = load_file('./lib/train_cut.txt')
fresq_label = call_feature(cut)

# 频繁词序列模式的结果写入文件
write_file(fresq_label, './lib/train_feature_fresq.txt')

 # 画出特征的直方图分布
figure_attribution(getnewList(fresq_label), 18)
# cut_label是频繁词序列模式得到的特征！！！