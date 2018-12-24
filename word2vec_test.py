import gensim
import numpy as np

word_vectors=gensim.models.KeyedVectors.load_word2vec_format('./tool/zhwiki_2017_03.sg_50d.word2vec',binary=False)
vocab = word_vectors.wv.vocab
print ('搜狗词向量语料加载，成功!')

'''显示所有词向量，并其存储在变量中'''
wordVectors = {}
wordsList = []
wordIndexVector = []
for word in vocab:
    wordVectors[word] = word_vectors[word] # 词语索引词向量（形式：dictionary）
    wordsList.append(word) # 数字索引词语（形式：list）
    wordIndexVector.append(word_vectors[word]) # 数字索引词向量（形式：list）


# 搜索文件，增加原本词向量库中没有的词语的向量和索引
def add_words(file_name):
    with open(file_name, "r", encoding='utf-8') as f:
        for line in f.readlines():
            row_words = line.split()
            for word in row_words:
                if word not in wordsList:
                    wordsList.append(word)
                    wordIndexVector.append(np.random.uniform(-1,1,size=50))


file_name_positive  = './lib/train_positive_cut.txt'
file_name_negtive  = './lib/train_negative_cut.txt'
add_words(file_name_positive)
print ('积极文本词语遍历，成功!')
add_words(file_name_negtive)
print ('消极文本词语遍历，成功!')


'''以npy文件格式保存变量，在下一个文件夹lstm中使用以下这两个变量'''
# 以npy文件格式存储变量（变量会以矩阵array形式保存，在下一次加载load该变量时，若是需要其它形式的变量，则需要转换变量形式）
np.save('./lib/wordsList.npy', wordsList) # 保存数字索引词语变量
print ('wordsList.npy保存，成功!')
np.save('./lib/wordIndexVector.npy', wordIndexVector) # 保存数字索引词向量变量
print ('wordIndexVector.npy保存，成功!')
