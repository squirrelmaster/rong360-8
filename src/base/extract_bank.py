import pandas as pd
import numpy as np


def get_feat():
    bank_detail_train = pd.read_csv('../data/train/bank_detail_train_80.csv', header=None)
    bank_detail_test_A = pd.read_csv('../data/test/bank_detail_test_A.csv', header=None)
    bank_detail_test_B = pd.read_csv('../data/test/bank_detail_test_B.csv', header=None)
    col_names = ['userid', 'tm_encode', 'trade_type', 'trade_amount', 'salary_tag']
    bank_detail_train.columns = col_names
    bank_detail_test_A.columns = col_names
    bank_detail_test_B.columns = col_names
    bank_detail = pd.concat([bank_detail_train, bank_detail_test_A, bank_detail_test_B])
    bank_detail['tm_encode'] = bank_detail['tm_encode'] // 86400

    loan_time_train = pd.read_csv('../data/train/loan_time_train_80.csv', header=None)
    loan_time_test_A = pd.read_csv('../data/test/loan_time_test_A.csv', header=None)
    loan_time_test_B = pd.read_csv('../data/test/loan_time_test_B.csv', header=None)
    loan_time = pd.concat([loan_time_train, loan_time_test_A, loan_time_test_B])
    loan_time.columns = ['userid', 'loan_time']
    loan_time['loan_time'] = loan_time['loan_time'] // 86400
    bank_detail = pd.merge(bank_detail, loan_time, on='userid')
    return bank_detail, loan_time


def extract_bank(bank_detail, loan_time):
    feature = loan_time
    d = bank_detail

    # ==========================放款前==========================
    gb1 = d[(d['tm_encode'] <= d['loan_time']) & d['trade_type'] == 0].groupby('userid', as_index=False)  # 收入
    gb2 = d[(d['tm_encode'] <= d['loan_time']) & d['trade_type'] == 1].groupby('userid', as_index=False)  # 支出
    gb3 = d[(d['tm_encode'] <= d['loan_time']) & d['salary_tag'] == 1].groupby('userid', as_index=False)  # 工资收入

    x1 = gb1['trade_amount'].agg({'b_t_t_0_size': np.size, 'b_t_t_0_sum': np.sum})
    x2 = gb2['trade_amount'].agg({'b_t_t_1_size': np.size, 'b_t_t_1_sum': np.sum})
    x3 = gb3['trade_amount'].agg({'b_s_t_1_size': np.size, 'b_s_t_1_sum': np.sum})

    feature = pd.merge(feature, x1, on='userid', how='left')
    feature = pd.merge(feature, x2, on='userid', how='left')
    feature = pd.merge(feature, x3, on='userid', how='left')

    feature['b_t_t_diff_size'] = feature['b_t_t_0_size'] - feature['b_t_t_1_size']
    feature['b_t_t_diff_sum'] = feature['b_t_t_0_sum'] - feature['b_t_t_1_sum']
    feature['b_n_s_t_diff_size'] = feature['b_t_t_0_size'] - feature['b_s_t_1_size']
    feature['b_n_s_t_diff_sum'] = feature['b_t_t_0_sum'] - feature['b_s_t_1_sum']
    feature['b_t_t_0xdiff_size'] = feature['b_t_t_0_size'] * feature['b_t_t_diff_size']
    feature['b_t_t_0xdiff_sum'] = feature['b_t_t_0_sum'] * feature['b_t_t_diff_sum']

    # ==========================放款后==========================
    gb1 = d[(d['tm_encode'] > d['loan_time']) & d['trade_type'] == 0].groupby('userid', as_index=False)  # 收入
    gb2 = d[(d['tm_encode'] > d['loan_time']) & d['trade_type'] == 1].groupby('userid', as_index=False)  # 支出
    gb3 = d[(d['tm_encode'] > d['loan_time']) & d['salary_tag'] == 1].groupby('userid', as_index=False)  # 工资收入

    x1 = gb1['trade_amount'].agg({'a_t_t_0_size': np.size, 'a_t_t_0_sum': np.sum})
    x2 = gb2['trade_amount'].agg({'a_t_t_1_size': np.size, 'a_t_t_1_sum': np.sum})
    x3 = gb3['trade_amount'].agg({'a_s_t_1_size': np.size, 'a_s_t_1_sum': np.sum})

    feature = pd.merge(feature, x1, on='userid', how='left')
    feature = pd.merge(feature, x2, on='userid', how='left')
    feature = pd.merge(feature, x3, on='userid', how='left')

    feature['a_t_t_diff_size'] = feature['a_t_t_0_size'] - feature['a_t_t_1_size']
    feature['a_t_t_diff_sum'] = feature['a_t_t_0_sum'] - feature['a_t_t_1_sum']
    feature['a_n_s_t_diff_size'] = feature['a_t_t_0_size'] - feature['a_s_t_1_size']
    feature['a_n_s_t_diff_sum'] = feature['a_t_t_0_sum'] - feature['a_s_t_1_sum']
    feature['a_t_t_0xdiff_size'] = feature['a_t_t_0_size'] * feature['a_t_t_diff_size']
    feature['a_t_t_0xdiff_sum'] = feature['a_t_t_0_sum'] * feature['a_t_t_diff_sum']

    print("11111111111111")
    print(feature.shape)
    return feature


def get_index():
    target_train = pd.read_csv('../data/train/overdue_train_80.csv', header=None)
    target_test_A = pd.read_csv('../data/test/overdue_test_A.csv', header=None)
    target_test_B = pd.read_csv('../data/test/overdue_test_B.csv', header=None)
    target = pd.concat([target_train, target_test_A, target_test_B])
    target.columns = ['userid', 'label']
    return target[['userid']]


if __name__ == '__main__':
    bank_detail, loan_time = get_feat()
    feat = extract_bank(bank_detail, loan_time)

    print("2222222222")
    print(feat.shape)
    feat.drop('loan_time', axis=1, inplace=True)
    feat = feat.drop_duplicates(subset=['userid'], keep='first')

    print("333333333333333")
    print(feat.shape)

    target = get_index()
    feat = pd.merge(target, feat, on='userid', how='left')

    print("44444444444444444")
    print(feat.shape)

    feat.to_csv("../cache/feature_bank", index=False)
