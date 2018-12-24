import numpy as np
import xgboost as xgb
import lightgbm as lgb

'''
0: 'Sentence', (text)
1: 'feature_enti_num', (num)
2: 'feature_fresq', (set)
3: 'feature_pos', (set)
4: 'feature_senti_num', (num)
5: 'feature_sentiment', (cat)
6: 'feature_slen', (num)
'''
CAT_FEAT = [5]
NUM_FEAT = [1, 4, 6]
SET_FEAT = [2, 3]

FEAT_TYPE = {0: 'text', 1: 'num', 2: 'set', 3: 'set', 4: 'num', 5: 'cat', 6: 'num'}


# parse a string into fields, skip quotations
def parse_feat(line):
    quota = False
    j = 0
    feats = []
    for i in range(len(line)):
        if line[i] == '\"':
            quota = not quota
        if line[i] == ',' and not quota:
            feat = line[j:i]
            feats.append(feat)
            j = i+1
    return feats + [line[j:]]


# load a csv file, use parse_feat() to convert format
def load_file(file_name):
    data = []
    with open(file_name, 'r') as fin:
        print('field_names:', fin.readline().strip().split(','))
        for line in fin:
            line = line.strip()
            data.append(parse_feat(line))
    return np.array(data)

def get_feat_name(field, feat_val):
    assert field != 'text'
    if FEAT_TYPE[field] == 'cat':
        return str(field) + ':' + feat_val
    elif FEAT_TYPE[field] == 'num':
        return str(field) + ':'
    elif FEAT_TYPE[field] == 'set':
        return [str(field) + ':' + fv for  fv in feat_val.split()]

def build_feat_map(data):
    feat_map = {}
    for i in range(len(FEAT_TYPE)):
        if FEAT_TYPE[i] == 'num':
            fn = get_feat_name(i, None)
            if fn not in feat_map:
                feat_map[fn] = len(feat_map)
            continue
        elif FEAT_TYPE[i] == 'text':
            continue
            
        feat = data[:, i]
        for f in feat:
            if FEAT_TYPE[i] == 'cat':
                fn = get_feat_name(i, f)
                if fn not in feat_map:
                    feat_map[fn] = len(feat_map)
            elif FEAT_TYPE[i] == 'set':
                for fn in get_feat_name(i, f):
                    if fn not in feat_map:
                        feat_map[fn] = len(feat_map)
    return feat_map

def to_float(x):
    if len(x):
        return float(x)
    return -1

def get_feat_id_val(field, feat_val):
    assert field != 'text'
    feat_name = get_feat_name(field, feat_val)
    if FEAT_TYPE[field] == 'cat':
        return feat_map[feat_name], 1
    elif FEAT_TYPE[field] == 'num':
        return feat_map[feat_name], to_float(feat_val)
    elif FEAT_TYPE[field] == 'set':
        return [feat_map[fn] for fn in feat_name], [1] * len(feat_name)

def to_libsvm(data):
    libsvm_data = []
    for d in data:
        libsvm_data.append([])
        for i in range(len(FEAT_TYPE)):
            if FEAT_TYPE[i] == 'cat' or FEAT_TYPE[i] == 'num':
                fv = get_feat_id_val(i, d[i])
                libsvm_data[-1].append(fv)
            elif FEAT_TYPE[i] == 'set':
                fvs = get_feat_id_val(i, d[i])
                for fv in zip(*fvs):
                    libsvm_data[-1].append(fv)
    return libsvm_data


train_data = load_file('./feature/train.csv')
# test_data = load_file('./feature/test.csv')

train_id, train_label, train_feat = train_data[:, 0], train_data[:, 1], train_data[:, 2:]
# test_id, test_feat = test_data[:, 0], test_data[:, 1:]

print('train_feat:\n', train_feat[0])
# print('test_feat:\n', test_feat[0])

train_feat[:, 0] = None
# test_feat[:, [1, 6]] = None

print('train_feat:\n', train_feat[0])
# print('test_feat:\n', test_feat[0])

feat_map = build_feat_map(train_feat)
# feat_map = build_feat_map(np.vstack([train_feat, test_feat]))

train_data = to_libsvm(train_feat)
# test_data = to_libsvm(test_feat)

np.random.seed(123)
rnd = np.random.random(len(train_data))
train_ind = np.where(rnd < 0.8)[0]
valid_ind = np.where(rnd >= 0.8)[0]

MAX_FEAT = ' %d:0\n' % len(feat_map)

flag = True
with open('./feature/train.svm', 'w') as fout:
    for i in train_ind:
        line = train_label[i]
        for fv in train_data[i]:
            line += ' {}:{}'.format(*fv)
        if flag:
            line += MAX_FEAT
            flag = False
        else:
            line += '\n'
        fout.write(line)
        
flag = True
with open('./feature/valid.svm', 'w') as fout:
    for i in valid_ind:
        line = train_label[i]
        for fv in train_data[i]:
            line += ' {}:{}'.format(*fv)
        if flag:
            line += MAX_FEAT
            flag = False
        else:
            line += '\n'
        fout.write(line)


dtrain = xgb.DMatrix('./feature/train.svm')
dtest = xgb.DMatrix('./feature/valid.svm')
param = {
    # learner params
    'booster': 'gbtree', # gbtree ot gblinear
    'nthread': 1, 
    'silent':1, 
    # tree params
    'eta':1, 
    'gamma': 0,
    'max_depth':4, 
    'subsample': 1,
    'lambda': 1,
    'alpha': 0,
    # learning params
    'objective':'binary:logistic', 
    'eval_metric': 'error', }
evallist = [(dtrain, 'train'), (dtest, 'eval')]
num_round = 50
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=5)
# bst.save_model('./model/xgb.model')
bst.dump_model('./model/xgb.dump.raw.txt')
# bst = xgb.Booster({'nthread': 4})  # init model
# bst.load_model('xgb.model')  # load data



dtrain = lgb.Dataset('./feature/train.svm')
dtest = lgb.Dataset('./feature/valid.svm')
param = {
    'objective':'binary',
    'boosting': 'gbdt',
    'num_threads': 1,
    'learning_rate': 1,
    'num_leaves': 31, 
    'max_depth': 9,
    'metric': 'binary_error',
    'lambda_l1': 0,
    'lambda_l2': 0,
    }

num_round = 100
bst = lgb.train(param, dtrain, num_round, valid_sets=[dtest], early_stopping_rounds=5)

bst.save_model('./model/model.txt')
# json_model = bst.dump_model()
# bst = lgb.Booster(model_file='model.txt') 