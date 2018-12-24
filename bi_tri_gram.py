from nltk.util import ngrams
import codecs # 写入txt文件

# 读取txt文件
def load_file(file_name):
    data = []
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
            for k in range(len(data[i][j])):
                fin.write(str(data[i][j][k])+" ")
            fin.write(",")
        fin.write("\n")
    fin.close()


# 生成bigram
def bi_gram(data):
    data_bi_gram = []
    for i in range(len(data)):
        data_bi_gram.append([])
        data_bigram = ngrams(data[i],2)
        for i in data_bigram:
            data_bi_gram[-1].append(list(i))
    return data_bi_gram


# 生成trigram
def tri_gram(data):
    data_tri_gram = []
    for i in range(len(data)):
        data_tri_gram.append([])
        if len(data[i]) > 2:
            data_trigram = ngrams(data[i],3)
        else:
            data_trigram = ngrams(data[i],2)
        for i in data_trigram:
            data_tri_gram[-1].append(list(i))
    return data_tri_gram


# 生成bigram,trigram
def bi_tri_gram(data):
    data_bi_tri_gram = []
    for i in range(len(data)):
        data_bi_tri_gram.append([])
        data_bigram = ngrams(data[i],2)
        data_tringram = ngrams(data[i],3)
        for i in data_bigram:
            data_bi_tri_gram[-1].append(list(i))
        for i in data_tringram:
            data_bi_tri_gram[-1].append(list(i))
    return data_bi_tri_gram


# 加载正、负文本的分词文件
train_data_positive_cut = load_file('./lib/train_positive_cut.txt')
train_data_negative_cut = load_file('./lib/train_negative_cut.txt')

# 调用函数，生成文本的bigram、trigram
# train_data_positive_bigram = bi_gram(train_data_positive_cut)
# train_data_positive_bi_trigram = bi_tri_gram(train_data_positive_cut)
train_data_positive_trigram = tri_gram(train_data_positive_cut)
train_data_negative_trigram = tri_gram(train_data_negative_cut)

# 将生成的bigram、trigram写入txt文件
# write_file(train_data_positive_bigram, './lib/train_data_positive_bigram.txt')
# write_file(train_data_positive_bi_trigram, './lib/train_data_positive_bi_trigram.txt')
write_file(train_data_positive_trigram, './lib/train_positive_trigram.txt')
write_file(train_data_negative_trigram, './lib/train_negative_trigram.txt')




