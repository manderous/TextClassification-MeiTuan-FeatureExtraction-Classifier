# PrefixSpan

import sys
import codecs # 写入txt文件
import numpy as np
# sys.setrecursionlimit(1000000) #例如这里设置为一百万
#import pdb
#pdb.set_trace()

PLACE_HOLDER = '_'

# 读取txt文件
def read(filename):
    S = []
    with open(filename, encoding='utf-8') as input:
        for line in input.readlines():
            elements = line.strip('\n').strip(',').split(',')
            s = []
            for e in elements:
                s.append(e.split())
            S.append(s)
    return S

# 写入txt文件
def write_file(patterns, file_name):
    fin = codecs.open(file_name, "w", "UTF-8")
    for i in patterns:
        fin.write("pattern:" + str(i.squence) + ", support:" + str(i.support))
        fin.write("\n")
    fin.close()

# 写入txt文件
def write_file1(dict, file_name):
    fin = codecs.open(file_name, "w", "UTF-8")
    for k,v in dict.items():
        fin.write("pattern:" + str(k) + ", dist:" + str(v))
        fin.write("\n")
    fin.close()

# 提取满足要求的频繁子序列Frequent subsequence保存成字典类型的变量，并存储在文件中
def frequent_sub(patterns, file_name):
    squence_support_dict = {}
    fin = codecs.open(file_name, "w", "UTF-8")
    for i in patterns:
        if len(i.squence) == 1:
            squence_support_dict.update({str(i.squence).replace('[','').replace(']','').replace("'",''):i.support})
            fin.write("pattern:" + str(i.squence).replace('[','').replace(']','').replace("'",'') + ", support:" + str(i.support) + "\n")
    fin.close()
    return squence_support_dict


# 合并3个字典，key的value组成一个列表，第1列是负例的supp100，第1列是负例的supp50，第3列是正例的supp50
def merge_dict(dict1, dict2, dict3, dict4):
    dict_hidden = {} #合并后的字典的初始化
    dict_hidden.update(dict1)
    dict_hidden.update(dict2)
    dict_hidden.update(dict3)
    dict_hidden.update(dict4)
    dictMerge_keyList = list(dict_hidden.keys())
    dict_merge = {}.fromkeys(dictMerge_keyList, [0,0,0,0])
    for k,v in dict1.items():
        dict_merge[k] = list(np.array(dict_merge[k])+np.array([v,0,0,0]))
    for k,v in dict2.items():
        dict_merge[k] = list(np.array(dict_merge[k])+np.array([0,v,0,0]))
    for k,v in dict3.items():
        dict_merge[k] = list(np.array(dict_merge[k])+np.array([0,0,v,0]))
    for k,v in dict4.items():
        dict_merge[k] = list(np.array(dict_merge[k])+np.array([0,0,0,v]))
    return dict_merge


class SquencePattern:
    def __init__(self, squence, support):
        self.squence = []
        for s in squence:
            self.squence.append(list(s))
        self.support = support

    def append(self, p):
        if p.squence[0][0] == PLACE_HOLDER:
            first_e = p.squence[0]
            first_e.remove(PLACE_HOLDER)
            self.squence[-1].extend(first_e)
            self.squence.extend(p.squence[1:])
        else:
            self.squence.extend(p.squence)
        self.support = min(self.support, p.support)


def prefixSpan(pattern, S, threshold):
    patterns = []
    f_list = frequent_items(S, pattern, threshold)
	
    for i in f_list:
        p = SquencePattern(pattern.squence, pattern.support)
        p.append(i)
        patterns.append(p)
        
        
        p_S = build_projected_database(S, p)
        p_patterns = prefixSpan(p, p_S, threshold)
        patterns.extend(p_patterns)
    return patterns


def frequent_items(S, pattern, threshold):
    items = {}
    _items = {}
    f_list = []
    if S is None or len(S) == 0:
        return []

    if len(pattern.squence) != 0:
        last_e = pattern.squence[-1]
    else:
        last_e = []
    for s in S:
        #class 1
        is_prefix = True
        for item in last_e:
            if item not in s[0]:
                is_prefix = False
                break
        if is_prefix and len(last_e) > 0:
            index = s[0].index(last_e[-1])
            if index < len(s[0]) - 1:
                for item in s[0][index + 1:]:
                    if item in _items:
                        _items[item] += 1
                    else:
                        _items[item] = 1

        #class 2
        if PLACE_HOLDER in s[0]:
            for item in s[0][1:]:
                if item in _items:
                    _items[item] += 1
                else:
                    _items[item] = 1
            s = s[1:]

        #class 3
        counted = []
        for element in s:
            for item in element:
                if item not in counted:
                    counted.append(item)
                    if item in items:
                        items[item] += 1
                    else:
                        items[item] = 1

    f_list.extend([SquencePattern([[PLACE_HOLDER, k]], v)
                    for k, v in _items.items()
                    if v >= threshold])
    f_list.extend([SquencePattern([[k]], v)
                   for k, v in items.items()
                   if v >= threshold])
    sorted_list = sorted(f_list, key=lambda p: p.support)
    return sorted_list  
    


def build_projected_database(S, pattern):
    """
    suppose S is projected database base on pattern's prefix,
    so we only need to use the last element in pattern to
    build projected database
    """
    p_S = []
    last_e = pattern.squence[-1]
    last_item = last_e[-1]
    for s in S:
        p_s = []
        for element in s:
            is_prefix = False
            if PLACE_HOLDER in element:
                if last_item in element and len(pattern.squence[-1]) > 1:
                    is_prefix = True
            else:
                is_prefix = True
                for item in last_e:
                    if item not in element:
                        is_prefix = False
                        break

            if is_prefix:
                e_index = s.index(element)
                i_index = element.index(last_item)
                if i_index == len(element) - 1:
                    p_s = s[e_index + 1:]
                else:
                    p_s = s[e_index:]
                    e = element[i_index:]
                    e[0] = PLACE_HOLDER
                    p_s[0] = e
                break
        if len(p_s) != 0:
            p_S.append(p_s)

    return p_S


def print_patterns(patterns):
    for p in patterns:
        print("pattern:{0}, support:{1}".format(p.squence, p.support))


if __name__ == "__main__":
    S_positive = read("./lib/train_positive_trigram.txt")
    patterns_positive = prefixSpan(SquencePattern([], sys.maxsize), S_positive, 50)
    # print_patterns(patterns) 显示出所有的频繁项，下面有把频繁项保存在txt文件中
    write_file(patterns_positive, "./lib/train_positive_prefixPattern.txt")
    squence_support50_dict_positive = frequent_sub(patterns_positive, './lib/train_positive_prefixFrequentSub_sup50.txt')
    
    patterns_positive = prefixSpan(SquencePattern([], sys.maxsize), S_positive, 25)
    squence_support25_dict_positive = frequent_sub(patterns_positive, './lib/train_positive_prefixFrequentSub_sup25.txt')
    
    S_negative = read("./lib/train_negative_trigram.txt")
    # 因为正负数据数量不平衡，所以支持度设置也不一样
    patterns_negative = prefixSpan(SquencePattern([], sys.maxsize), S_negative, 100)
    write_file(patterns_negative, "./lib/train_negative_prefixPattern.txt")
    squence_support100_dict_negative = frequent_sub(patterns_negative, './lib/train_negative_prefixFrequentSub_sup100.txt')
    
    patterns_negative = prefixSpan(SquencePattern([], sys.maxsize), S_negative, 50)
    squence_support50_dict_negative = frequent_sub(patterns_negative, './lib/train_negative_prefixFrequentSub_sup50.txt')
    
    
    # 合并正例和负例的频繁词序列的字典
    squence_support_dict = merge_dict(squence_support100_dict_negative, squence_support50_dict_negative, squence_support50_dict_positive, squence_support25_dict_positive)
    
    squence_support_dist_dict = {} # 计算词典中每个频繁词序列的区分度dist（初始化）
    squence_support_dist_dict_filter = {} # 筛选出区分度dist>=0.85的频繁词序列重新组成词典（初始化）
    for k,v in squence_support_dict.items():
        if v[1] == 0: # 这里做了一个近似，如果支持度小于50，直接令其支持度为50（负例）
            v[1] = 50
        if v[3] == 0: # 这里做了一个近似，如果支持度小于25，直接令其支持度为25
            v[3] = 25
        dist = max(v[1]*0.5,v[3])/(v[1]*0.5+v[3]) # 计算每个频繁词序列的区分度dist（正例）
        squence_support_dist_dict.update({k:dist})
        if dist >= 0.85:
            squence_support_dist_dict_filter.update({k:dist})
    write_file1(squence_support_dist_dict_filter, "./lib/train_squence_dict.txt")









