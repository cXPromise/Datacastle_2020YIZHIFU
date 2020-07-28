# coding: utf-8
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import gc
from catboost import Pool, CatBoostClassifier
from gensim.models import Word2Vec
import os
warnings.filterwarnings('ignore')


def load_dataset(DATA_PATH):
    train_label = pd.read_csv(DATA_PATH+'train_label.csv')
    train_base = pd.read_csv(DATA_PATH+'train_base.csv')
    test_base = pd.read_csv(DATA_PATH+'test_a_base.csv')

    train_op = pd.read_csv(DATA_PATH+'train_op.csv')
    train_trans = pd.read_csv(DATA_PATH+'train_trans.csv')
    test_op = pd.read_csv(DATA_PATH+'test_a_op.csv')
    test_trans = pd.read_csv(DATA_PATH+'test_a_trans.csv')

    return train_label, train_base, test_base, train_op, train_trans, test_op, test_trans

def data_preprocess(DATA_PATH):
    train_label, train_base, test_base, train_op, train_trans, test_op, test_trans = load_dataset(DATA_PATH=DATA_PATH)
    #
    train_df = train_base.copy()
    test_df = test_base.copy()
    train_df = train_label.merge(train_df, on=['user'], how='left')
    del train_base, test_base
    #
    op_df = pd.concat([train_op, test_op], axis=0, ignore_index=True)
    trans_df = pd.concat([train_trans, test_trans], axis=0, ignore_index=True)
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    del train_op, test_op, train_df, test_df
    #
    op_df['days_diff'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[0]))
    trans_df['days_diff'] = trans_df['tm_diff'].apply(lambda x: int(x.split(' ')[0]))
    op_df['timestamp'] = op_df['tm_diff'].apply(lambda x: transform_time(x))
    trans_df['timestamp'] = trans_df['tm_diff'].apply(lambda x: transform_time(x))
    op_df['hour'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[0]))
    trans_df['hour'] = trans_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[0]))
    trans_df['week'] = trans_df['days_diff'].apply(lambda x: x % 7)
    #
    trans_df = trans_df.sort_values(by=['user', 'timestamp'])
    op_df = op_df.sort_values(by=['user', 'timestamp'])
    #
    trans_df.reset_index(inplace=True, drop=True)
    op_df.reset_index(inplace=True, drop=True)

    gc.collect()
    return data, op_df, trans_df

def gen_user_amount_features(trans):
    group_df = trans.groupby(['user'])['amount'].agg({'user_amount_mean': 'mean',
                                                      'user_amount_std': 'std',
                                                      'user_amount_max': 'max',
                                                      'user_amount_min': 'min',
                                                      'user_amount_sum': 'sum',
                                                      'user_amount_skew': 'skew',
                                                      'user_amount_med': 'median',
                                                      'user_amount_cnt': 'count',
                                                      }).reset_index()
    return group_df

def gen_user_day_amount_features(trans):
    group_df = trans.groupby(['user', 'days_diff'])['amount'].agg({'user_amount_sum': 'sum'}).reset_index()
    group_df = group_df.groupby(['user'])['user_amount_sum'].agg({'user_amount_day_max': 'max',
                                                                  'user_amount_day_std': 'std',
                                                                  'user_amount_days_med': 'median'}).reset_index()
    return group_df

def gen_user_group_amount_features(trans, value):
    group_df = trans.pivot_table(index='user',
                                     columns=value,
                                     values='amount',
                                     dropna=False,
                                     aggfunc=['count', 'sum']).fillna(0)
    group_df.columns = ['user_{}_{}_amount_{}'.format(value, f[1], f[0]) for f in group_df.columns]
    group_df.reset_index(inplace=True)

    return group_df

def gen_user_mode_features(df, value, prefix):
    group_df = df.groupby(['user', value])['tm_diff'].agg({'cnt': 'count'}).reset_index()
    group_df['rank'] = group_df.groupby(['user'])['cnt'].rank(ascending=False, method='first')
    result = group_df[['user']].drop_duplicates()
    for i in range(1, 2):
        tmp = group_df[group_df['rank']==i].copy()
        del tmp['rank']
        tmp.columns = ['user', 'user_{}_mode_{}_{}'.format(prefix, i, value), 'user_mode_{}_{}_{}_cnt'.format(prefix, i, value)]
        result = result.merge(tmp, on=['user'], how='left')
    return result

def gen_user_window_amount_features(trans, window):
    group_df = trans[trans['days_diff']>window].groupby('user')['amount'].agg({'user_amount_mean_{}d'.format(window): 'mean',
                                                                              'user_amount_std_{}d'.format(window): 'std',
                                                                              'user_amount_max_{}d'.format(window): 'max',
                                                                              'user_amount_min_{}d'.format(window): 'min',
                                                                              'user_amount_sum_{}d'.format(window): 'sum',
                                                                              'user_amount_skew_{}d'.format(window): 'skew',
                                                                              'user_amount_med_{}d'.format(window): 'median',
                                                                              'user_amount_cnt_{}d'.format(window): 'count',
                                                                               }).reset_index()
    return group_df

def gen_user_window_count_features(df, window, prefix):
    group_df = df[df['days_diff']>window].groupby('user')['tm_diff'].agg({'user_{}_cnt_{}d'.format(prefix, window): 'count'}).reset_index()
    return group_df

def gen_user_nunique_features(df, value, prefix):
    group_df = df.groupby(['user'])[value].agg({'user_{}_{}_nuniq'.format(prefix, value): 'nunique'}).reset_index()
    return group_df

def gen_user_null_features(df, value, prefix):
    df['is_null'] = 0
    df.loc[df[value].isnull(), 'is_null'] = 1

    group_df = df.groupby(['user'])['is_null'].agg({'user_{}_{}_null_cnt'.format(prefix, value): 'sum',
                                                    'user_{}_{}_null_ratio'.format(prefix, value): 'mean'}).reset_index()
    return group_df

def gen_user_tfidf_features(df, value):
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    group_df.columns = ['user', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_vec = TfidfVectorizer()
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2020)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_tfidf_{}_{}'.format(value, i) for i in range(10)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df

def gen_user_countvec_features(df, value):
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    group_df.columns = ['user', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_vec = CountVectorizer()
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2020)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_countvec_{}_{}'.format(value, i) for i in range(10)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df

def gen_user_ip_features(df, value, prefix):
    group_df = df.groupby([value])['days_diff'].agg({'cnt': 'nunique'}).reset_index()
    group_df = df.merge(group_df, on=[value], how='left')
    group_df = group_df.groupby(['user'])['cnt'].agg({'user_{}_{}_nuniq_mean'.format(prefix, value): 'mean',
                                                      'user_{}_{}_nuniq_skew'.format(prefix, value): 'skew',
                                                      'user_{}_{}_nuniq_std'.format(prefix, value): 'std',
                                                      'user_{}_{}_nuniq_min'.format(prefix, value): 'min',
                                                      'user_{}_{}_nuniq_max'.format(prefix, value): 'max',}).reset_index()
    return group_df


def cal_sim_cnt(x, y):
    if str(x)!='nan' and str(y)!='nan':
        return len(set(x)&set(y))
    else:
        return np.nan

def cal_sim_ratio(x, y):
    if str(x)!='nan' and str(y)!='nan':
        return 1.0*len(set(x)&set(y))/len(set(x).union(set(y)))
    else:
        return np.nan

def gen_op_trans_ip_sim(df, op, trans, value):
    tmp1 = op[~op[value].isnull()].copy()
    tmp2 = trans[~trans[value].isnull()].copy()
    group1 = tmp1.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    group1.columns = ['user', 'op']
    group2 = tmp2.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    group2.columns = ['user', 'trans']

    df = df.merge(group1, on=['user'], how='left')
    df = df.merge(group2, on=['user'], how='left')

    df['op2trans_{}_sim_cnt'.format(value)] = df.apply(lambda x: cal_sim_cnt(x['op'], x['trans']), axis=1)
    df['op2trans_{}_sim_ratio'.format(value)] = df.apply(lambda x: cal_sim_ratio(x['op'], x['trans']), axis=1)
    del df['op'], df['trans']

    return df


def transform_time(x):
    day = int(x.split(' ')[0])
    hour = int(x.split(' ')[2].split('.')[0].split(':')[0])
    minute = int(x.split(' ')[2].split('.')[0].split(':')[1])
    second = int(x.split(' ')[2].split('.')[0].split(':')[2])
    return 86400*day+3600*hour+60*minute+second

def gen_user_timediff_features(df, prefix):
    tmp = df.copy()
    tmp = tmp.sort_values(by=['user', 'timestamp'])
    tmp['shift_value'] = tmp.groupby(['user'])['timestamp'].shift(-1)
    tmp['timedelta'] = tmp['shift_value'] - tmp['timestamp']
    group_df = tmp.groupby(['user'])['timedelta'].agg({'user_{}_timedelta_mean'.format(prefix): 'mean',
                                                       'user_{}_timedelta_max'.format(prefix): 'max',
                                                       'user_{}_timedelta_min'.format(prefix): 'min',
                                                       'user_{}_timedelta_std'.format(prefix): 'std',
                                                       'user_{}_timedelta_skew'.format(prefix): 'skew',
                                                       'user_{}_timedelta_med'.format(prefix): 'median',}).reset_index()
    return group_df


def gen_user_amount_diff_features(df, prefix):
    tmp = df.copy()
    tmp = tmp.sort_values(by=['user', 'timestamp'])
    tmp['shift_value'] = tmp.groupby(['user'])['amount'].shift(-1)
    tmp['timedelta'] = tmp['shift_value'] - tmp['amount']
    group_df = tmp.groupby(['user'])['timedelta'].agg({'user_{}_amount_diff_mean'.format(prefix): 'mean',
                                                       'user_{}_amount_diff_max'.format(prefix): 'max',
                                                       'user_{}_amount_diff_min'.format(prefix): 'min',
                                                       'user_{}_amount_diff_std'.format(prefix): 'std',
                                                       'user_{}_amount_diff_skew'.format(prefix): 'skew',
                                                       'user_{}_amount_diff_med'.format(prefix): 'median',}).reset_index()
    return group_df


def gen_user_hour_amount_features(df):
    group_df = df.groupby(['user', 'day_diff', 'hour'])['amount'].agg({'mean',
                                                                       'sum',
                                                                       'count'}).reset_index()
    agg_fun = {'mean': ['mean', 'max'],
               'sum': ['mean', 'max'],
               'count': ['mean', 'max']}
    group_df = group_df.groupby(['user']).agg(agg_fun)
    group_df.columns = ['{}_{}_day_hour'.format(f[0], f[1]) for f in group_df.columns]
    group_df.reset_index(inplace=True)
    return group_df

def gen_magic_features(df, value):
    group_df = df.groupby(value)['days_diff'].agg({'max', 'min', 'nunique'}).reset_index()
    group_df['range'] = group_df['max'] - group_df['min']
    tmp = df.merge(group_df, on=value, how='left')
    agg_fun = {'range': ['mean', 'max', 'std'],
               'nunique': ['mean', 'max', 'std']}
    group_df = tmp.groupby(['user']).agg(agg_fun)
    group_df.columns = ['{}_{}_{}_days'.format(f[0], f[1], value) for f in group_df.columns]
    group_df.reset_index(inplace=True)
    return group_df

def gen_word2vec_feature(df, groupby, target, size, window, prefix):
    path = '../cache/{}_{}_{}_w2v.csv'.format('_'.join(groupby), target, prefix)
    print(path)
    if os.path.exists(path):
        df_w2v = pd.read_csv(path)
    else:
        df_bag = df[groupby+[target]].copy()
        df_bag[target] = df_bag[target].astype(str)
        df_bag[target].fillna('-1', inplace=True)
        df_bag = df_bag.groupby(groupby, as_index=False)[target].agg({'list': (lambda x: list(x))}).reset_index()
        doc_list = list(df_bag['list'].values)
        w2v = Word2Vec(doc_list, size=size, window=window, min_count=1, workers=32)
        vocab_keys = list(w2v.wv.vocab.keys())
        w2v_array = []
        for v in vocab_keys:
            w2v_array.append(list(w2v.wv[v]))
        df_w2v = pd.DataFrame()
        df_w2v['vocab_keys'] = vocab_keys
        df_w2v = pd.concat([df_w2v, pd.DataFrame(w2v_array)], axis=1)
        df_w2v.columns = [target] + ['w2v_%s_%s_%d' % (prefix, target, x) for x in range(size)]
        df_w2v.to_csv(path, index=False)
        print('df_w2v:' + str(df_w2v.shape))
    return df_w2v


def kfold_stats_feature(train, test, feats, k):
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = ['label']
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_trn = train.iloc[trn_idx]
                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = train[f].mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            test[colname] = None
            order_label = train.groupby([feat])[f].mean()
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = train[f].mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test

def gen_features(df, op, trans):
    df.drop(['service3_level'], axis=1, inplace=True)

    # base
    df['product7_fail_ratio'] = df['product7_fail_cnt'] / df['product7_cnt']

    # trans
    df = df.merge(gen_user_amount_features(trans), on=['user'], how='left')
    for col in tqdm(['days_diff', 'platform', 'tunnel_in', 'tunnel_out', 'type1', 'type2', 'ip', 'ip_3']):
        df = df.merge(gen_user_nunique_features(df=trans, value=col, prefix='trans'), on=['user'], how='left')
    df['user_amount_per_days'] = df['user_amount_sum'] / df['user_trans_days_diff_nuniq']
    df['user_amount_per_cnt'] = df['user_amount_sum'] / df['user_amount_cnt']
    df = df.merge(gen_user_group_amount_features(trans, 'platform'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(trans, 'type1'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(trans, 'type2'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(trans, 'tunnel_in'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(trans, 'tunnel_out'), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(trans, 27), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(trans, 23), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(trans, 15), on=['user'], how='left')
    df = df.merge(gen_user_null_features(trans, 'ip', 'trans'), on=['user'], how='left')

    # op
    df = df.merge(gen_user_tfidf_features(df=op, value='op_mode'), on=['user'], how='left')
    df = df.merge(gen_user_tfidf_features(df=op, value='op_type'), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_mode'), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_type'), on=['user'], how='left')

    group_df = trans[trans['type1']=='45a1168437c708ff'].groupby(['user'])['days_diff'].agg({'user_type1_45a1168437c708ff_min_day': 'min',
                                                                                             # 'user_type1_45a1168437c708ff_max_day': 'max',
                                                                                             # 'user_type1_45a1168437c708ff_cnt_day': 'nunique'
                                                                                             }).reset_index()
    # group_df['user_type1_45a1168437c708ff_day_range'] = group_df['user_type1_45a1168437c708ff_max_day'] - group_df['user_type1_45a1168437c708ff_min_day']
    df = df.merge(group_df, on=['user'], how='left')


    df['city_count'] = df.groupby(['city'])['user'].transform('count')
    df['province_count'] = df.groupby(['province'])['user'].transform('count')
    # df = df.merge(gen_user_timediff_features(df=trans[trans['type1']=='45a1168437c708ff'], prefix='trans'), on=['user'], how='left')


    # user_ = df[['user']].drop_duplicates()
    # df = df.merge(gen_op_trans_ip_sim(user_, op, trans, 'ip'), on=['user'], how='left')
    # group_df = trans.groupby(['user', 'days_diff', 'amount'])['tm_diff'].count().reset_index()
    # group_df = group_df[group_df['tm_diff']>=2].groupby(['user'])['tm_diff'].agg({'user_same_amount_cnt': 'count'}).reset_index()
    # df = df.merge(group_df, on=['user'], how='left')
    # # df = df.merge(gen_user_group_amount_features(trans, 'week'), on=['user'], how='left')
    # print(df.head())
    # df = df.merge(gen_user_amount_diff_features(trans, 'trans'), on=['user'], how='left')
    # df = df.merge(gen_magic_features(op, 'op_device'), on=['user'], how='left')
    # for col in tqdm(['op_device']):
    #     df = df.merge(gen_user_nunique_features(df=op, value=col, prefix='op'), on=['user'], how='left')
    # df['acc_count/ip_nunique'] = df['acc_count'] / df['user_trans_ip_nuniq']
    # df['acc_count/ip3_nunique'] = df['acc_count'] / df['user_trans_ip_3_nuniq']
    # df['acc_count/op_device_nunique'] = df['acc_count'] / df['user_op_op_device_nuniq']

    # # # w2v
    # for i in trans['type1'].unique():
    #     if str(i) != 'nan':
    #         df['user_type1_{}_amount_ratio'.format(i)] = df['user_type1_{}_amount_count'.format(i)].fillna(0) / (df['user_amount_cnt'].fillna(0) + 1)
    #         df['user_type1_{}_amount_ratio2'.format(i)] = df['user_type1_{}_amount_sum'.format(i)].fillna(0) / (df['user_amount_sum'].fillna(0) + 1)
    #
    # for i in trans['type2'].unique():
    #     if str(i) != 'nan':
    #         df['user_type2_{}_amount_ratio'.format(i)] = df['user_type2_{}_amount_count'.format(i)].fillna(0) / (df['user_amount_cnt'].fillna(0) + 1)
    #         df['user_type2_{}_amount_ratio2'.format(i)] = df['user_type2_{}_amount_sum'.format(i)].fillna(0) / (df['user_amount_sum'].fillna(0) + 1)

    # user_trans_ip_w2v = gen_word2vec_feature(df=trans, groupby=['user'], target='amount', window=5, size=10, prefix='trans')
    # user_trans_ip_w2v['amount'] = user_trans_ip_w2v['amount'].astype(int)
    # tmp = trans.merge(user_trans_ip_w2v, on=['amount'], how='left')
    # w2v_cols = [f for f in user_trans_ip_w2v.columns if f not in ['amount']]
    # group_df = tmp.groupby(['user'])[w2v_cols].mean().reset_index()
    # df = df.merge(group_df, on=['user'], how='left')

    # user_op_mode_w2v = gen_word2vec_feature(df=op, groupby=['user'], target='op_type', window=5, size=10, prefix='op')
    # tmp = op.merge(user_op_mode_w2v, on=['op_type'], how='left')
    # w2v_cols = [f for f in user_op_mode_w2v.columns if f not in ['op_type']]
    # group_df = tmp.groupby(['user'])[w2v_cols].mean().reset_index()
    # df = df.merge(group_df, on=['user'], how='left')

    # df = df.merge(gen_user_mode_features(df=op, value='op_mode', prefix='op'), on=['user'], how='left')
    # df = df.merge(gen_user_mode_features(df=op, value='op_type', prefix='op'), on=['user'], how='left')

    # df = df.merge(gen_user_window_count_features(df=op, window=3, prefix='op'), on=['user'], how='left')
    # df = df.merge(gen_user_window_count_features(df=op, window=7, prefix='op'), on=['user'], how='left')
    # df = df.merge(gen_user_window_count_features(df=op, window=15, prefix='op'), on=['user'], how='left')

    # df['user_'] = df['user'].apply(lambda x: int(x.split('_')[-1]))

    # LabelEncoder
    cat_cols = []
    for col in tqdm([f for f in df.select_dtypes('object').columns if f not in ['user']]):
        le = LabelEncoder()
        df[col].fillna('-1', inplace=True)
        df[col] = le.fit_transform(df[col])
        cat_cols.append(col)

    return df

def lgb_model(train, target, test, k):
    feats = [f for f in train.columns if f not in ['user', 'label', 'x', 'y']]
    print('Current num of features:', len(feats))
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)
    oof_probs = np.zeros(train.shape[0])
    output_preds = 0
    offline_score = []
    feature_importance_df = pd.DataFrame()
    parameters = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'nthread': 8
    }

    for i, (train_index, test_index) in enumerate(folds.split(train, target)):
        train_y, test_y = target[train_index], target[test_index]
        train_X, test_X = train[feats].iloc[train_index, :], train[feats].iloc[test_index, :]

        dtrain = lgb.Dataset(train_X,
                             label=train_y)
        dval = lgb.Dataset(test_X,
                           label=test_y)
        lgb_model = lgb.train(
                parameters,
                dtrain,
                num_boost_round=5000,
                valid_sets=[dval],
                early_stopping_rounds=100,
                verbose_eval=100,
                # categorical_feature=['card_a_cnt', 'card_b_cnt', 'card_c_cnt']
        )
        oof_probs[test_index] = lgb_model.predict(test_X[feats], num_iteration=lgb_model.best_iteration)
        offline_score.append(lgb_model.best_score['valid_0']['auc'])
        output_preds += lgb_model.predict(test[feats], num_iteration=lgb_model.best_iteration)/folds.n_splits
        print(offline_score)
        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = lgb_model.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('OOF-MEAN-AUC:%.6f, OOF-STD-AUC:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    print('feature importance:')
    print(feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False).head(15))

    return output_preds, oof_probs, np.mean(offline_score)

def cbt_model(train, target, test, k):
    feats = [f for f in train.columns if f not in ['user', 'label']]
    print('Current num of features:', len(feats))
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)
    oof_probs = np.zeros(train.shape[0])
    output_preds = 0
    offline_score = []
    catboost_params = {
        'iterations': 10000,
        'learning_rate': 0.05,
        'eval_metric': 'AUC',
        'task_type': 'GPU',
        'early_stopping_rounds': 100,
        'use_best_model': True,
        'verbose': 100,
    }

    for i, (train_index, test_index) in enumerate(folds.split(train, target)):
        train_y, test_y = target[train_index], target[test_index]
        train_X, test_X = train[feats].iloc[train_index, :], train[feats].iloc[test_index, :]

        train_pool = Pool(train_X, train_y,
                          cat_features=[u'sex', u'provider', u'level', u'verified', u'regist_type',
       u'agreement1', u'agreement2', u'agreement3', u'agreement4', u'province',
       u'city', u'balance', u'balance_avg', u'balance1', u'balance1_avg',
       u'balance2', u'balance2_avg', u'service3', u'service3_level',
       u'product1_amount', u'product2_amount', u'product3_amount',
       u'product4_amount', u'product5_amount', u'product6_amount']
                          )
        valid_pool = Pool(test_X, test_y,
                          cat_features=[u'sex', u'provider', u'level', u'verified', u'regist_type',
       u'agreement1', u'agreement2', u'agreement3', u'agreement4', u'province',
       u'city', u'balance', u'balance_avg', u'balance1', u'balance1_avg',
       u'balance2', u'balance2_avg', u'service3', u'service3_level',
       u'product1_amount', u'product2_amount', u'product3_amount',
       u'product4_amount', u'product5_amount', u'product6_amount']
                          )

        model = CatBoostClassifier(**catboost_params)
        model.fit(train_pool, eval_set=valid_pool)

        oof_probs[test_index] = model.predict_proba(test_X)[:, 1]
        offline_score.append(model.best_score_['validation']['AUC'])
        output_preds += model.predict_proba(test[feats])[:, 1] / folds.n_splits

    print('线下K折AUC: %.5f' % (np.mean(offline_score)))
    return output_preds, oof_probs, np.mean(offline_score)



if __name__ == '__main__':
    DATA_PATH = '../data/'
    print('读取数据...')
    data, op_df, trans_df = data_preprocess(DATA_PATH=DATA_PATH)

    print('开始特征工程...')
    data = gen_features(data, op_df, trans_df)
    data['x'] = data['city'].map(str) + '_' + data['level'].map(str)
    data['y'] = data['city'].map(str) + '_' + data['balance_avg'].map(str)
    data['z'] = data['city'].map(str) + '_' + data['balance_avg'].map(str)

    print('开始模型训练...')
    train = data[~data['label'].isnull()].copy()
    target = train['label']
    test = data[data['label'].isnull()].copy()

    target_encode_cols =  [
                         'province',
                         # 'service1_amt',
                         # 'acc_count',
                         # 'login_cnt_period1',
                         # 'login_cnt_period2',
                         # 'ip_cnt',
                         # 'login_cnt_avg',
                         # 'login_days_cnt',
                         'city',
                        'x',
    ]
    train, test = kfold_stats_feature(train, test, target_encode_cols, 5)

    lgb_preds, lgb_oof, lgb_score = lgb_model(train=train, target=target, test=test, k=5)
    # cbt_preds, cbt_oof, cbt_score = cbt_model(train=train, target=target, test=test, k=5)

    sub_df = test[['user']].copy()
    sub_df['prob'] = lgb_preds
    sub_df.to_csv('../submission/base_sub.csv', index=False)

