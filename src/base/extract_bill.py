import pandas as pd
import numpy as np


def get_feat():
    bill_detail_train = pd.read_csv('../data/train/bill_detail_train_80.csv', header=None)
    bill_detail_test_A = pd.read_csv('../data/test/bill_detail_test_A.csv', header=None)
    bill_detail_test_B = pd.read_csv('../data/test/bill_detail_test_B.csv', header=None)
    col_names = ['userid', 'tm_encode_3', 'bank_id', 'prior_account', 'prior_repay', 'credit_limit', 'account_balance',
                 'minimun_repay', 'consume_count', 'account', 'adjust_account', 'circulated_interest',
                 'avaliable_balance', 'cash_limit', 'repay_state']
    bill_detail_train.columns = col_names
    bill_detail_test_A.columns = col_names
    bill_detail_test_B.columns = col_names
    bill_detail = pd.concat([bill_detail_train, bill_detail_test_A, bill_detail_test_B])
    bill_detail['tm_encode_3'] = bill_detail['tm_encode_3'] // 86400

    loan_time_train = pd.read_csv('../data/train/loan_time_train_80.csv', header=None)
    loan_time_test_A = pd.read_csv('../data/test/loan_time_test_A.csv', header=None)
    loan_time_test_B = pd.read_csv('../data/test/loan_time_test_B.csv', header=None)
    loan_time = pd.concat([loan_time_train, loan_time_test_A, loan_time_test_B])
    loan_time.columns = ['userid', 'loan_time']
    loan_time['loan_time'] = loan_time['loan_time'] // 86400
    bill_detail = pd.merge(bill_detail, loan_time, on='userid')
    return bill_detail, loan_time


def extract_bill_1(bill_detail, loan_time):
    feature1 = loan_time
    b_loan = bill_detail[bill_detail['tm_encode_3'] <= bill_detail['loan_time']]
    feature1 = bill_util(b_loan, 'prior_account', feature1)
    feature1 = bill_util(b_loan, 'prior_repay', feature1)
    feature1['bill_diff'] = feature1['prior_account_sum'] - feature1['prior_repay_sum']

    feature2 = loan_time
    a_loan = bill_detail[bill_detail['tm_encode_3'] > bill_detail['loan_time']]
    feature2 = bill_util(a_loan, 'prior_account', feature2)
    feature2 = bill_util(a_loan, 'prior_repay', feature2)
    feature2['bill_diff'] = feature2['prior_account_sum'] - feature2['prior_repay_sum']

    feature = pd.merge(feature1, feature2, on=['userid', 'loan_time'], suffixes=('_b', '_a'))
    print("11111111111111")
    print(feature.shape)
    return feature


def bill_util(data, col, feature):
    temp = data.groupby('userid', as_index=False)[col]
    x1 = temp.apply(lambda x: x.where(x < 0).count())
    x2 = temp.apply(lambda x: x.where(x == 0).count())
    x3 = temp.agg({col + '_sum': np.sum})
    x3[col + '_less0'] = x1
    x3[col + '_equ0'] = x2
    feature = pd.merge(feature, x3, how='left')
    return feature


def extract_bill_2(bill_detail, loan_time):
    d = bill_detail
    d1 = d[(d['prior_account'] <= 0) | (d['prior_repay'] <= 0)].index.tolist()
    d = d.drop(d1, axis=0)

    # 放款前数据提取特征
    data = d[d['tm_encode_3'] <= d['loan_time']]
    gb = data.groupby(['userid', 'tm_encode_3', 'bank_id'], as_index=False)
    x1 = gb['prior_account'].agg({'b_p_a_sum': np.sum, 'b_p_a_max': np.max})
    x2 = gb['prior_repay'].agg({'b_p_r_sum': np.sum, 'b_p_r_max': np.max})
    x3 = gb['circulated_interest'].agg({'b_c_i_max': np.max})
    x4 = gb['consume_count'].agg({'b_c_c_max': np.max})

    gb1 = x1.groupby('userid', as_index=False)
    gb2 = x2.groupby('userid', as_index=False)
    gb3 = x3.groupby('userid', as_index=False)
    gb4 = x4.groupby('userid', as_index=False)

    x11 = gb1['b_p_a_sum'].agg({'b_p_a_sum_sum': np.sum, 'b_p_a_sum_size': np.size})
    x12 = gb1['b_p_a_max'].agg({'b_p_a_max_sum': np.sum})

    x21 = gb2['b_p_r_sum'].agg({'b_p_r_sum_sum': np.sum, 'b_p_r_sum_size': np.size})
    x22 = gb2['b_p_r_max'].agg({'b_p_r_max_sum': np.sum})

    x31 = gb3['b_c_i_max'].agg({'b_c_i_max_sum': np.sum})
    x41 = gb4['b_c_c_max'].agg({'b_c_c_max_sum': np.sum})

    feature = x11
    feature = pd.merge(feature, x12, on='userid', how='left')
    feature = pd.merge(feature, x21, on='userid', how='left')
    feature = pd.merge(feature, x22, on='userid', how='left')
    feature = pd.merge(feature, x31, on='userid', how='left')
    feature = pd.merge(feature, x41, on='userid', how='left')

    # 放款后数据提取特征
    data = d[d['tm_encode_3'] > d['loan_time']]
    gb = data.groupby(['userid', 'tm_encode_3', 'bank_id'], as_index=False)
    x1 = gb['prior_account'].agg({'a_p_a_sum': np.sum, 'a_p_a_max': np.max})
    x2 = gb['prior_repay'].agg({'a_p_r_sum': np.sum, 'a_p_r_max': np.max})
    x3 = gb['circulated_interest'].agg({'a_c_i_max': np.max})
    x4 = gb['consume_count'].agg({'a_c_c_max': np.max})

    gb1 = x1.groupby('userid', as_index=False)
    gb2 = x2.groupby('userid', as_index=False)
    gb3 = x3.groupby('userid', as_index=False)
    gb4 = x4.groupby('userid', as_index=False)

    x11 = gb1['a_p_a_sum'].agg({'a_p_a_sum_sum': np.sum, 'a_p_a_sum_size': np.size})
    x12 = gb1['a_p_a_max'].agg({'a_p_a_max_sum': np.sum})

    x21 = gb2['a_p_r_sum'].agg({'a_p_r_sum_sum': np.sum, 'a_p_r_sum_size': np.size})
    x22 = gb2['a_p_r_max'].agg({'a_p_r_max_sum': np.sum})

    x31 = gb3['a_c_i_max'].agg({'a_c_i_max_sum': np.sum})
    x41 = gb4['a_c_c_max'].agg({'a_c_c_max_sum': np.sum})

    feature = pd.merge(feature, x11, on='userid', how='left')
    feature = pd.merge(feature, x12, on='userid', how='left')
    feature = pd.merge(feature, x21, on='userid', how='left')
    feature = pd.merge(feature, x22, on='userid', how='left')
    feature = pd.merge(feature, x31, on='userid', how='left')
    feature = pd.merge(feature, x41, on='userid', how='left')
    print("2222222222222")
    print(feature.shape)
    return feature


def extract_bill_3(bill_detail, loan_time):
    # 刷爆次数
    d = bill_detail
    gb = d[d['credit_limit'] < d['account_balance']].groupby('userid', as_index=False)
    x1 = gb['tm_encode_3'].apply(lambda x: np.unique(x).size)
    x2 = gb['tm_encode_3'].agg({'ex_size': np.size})
    x2['uniq_ex_size'] = x1

    feature = x2

    # 银行持卡数
    gb = d.groupby('userid', as_index=False)
    x1 = gb['bank_id'].apply(lambda x: np.unique(x).size)
    x2 = gb['bank_id'].agg({'h_card': np.size})
    x2['uniq_h_card'] = x1
    feature = pd.merge(feature, x2, on='userid', how='left')
    print("333333333333")
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
    bill_detail, loan_time = get_feat()
    feat_1 = extract_bill_1(bill_detail, loan_time)
    feat_2 = extract_bill_2(bill_detail, loan_time)
    feat_3 = extract_bill_3(bill_detail, loan_time)
    feat = pd.merge(feat_1, feat_2, on='userid', how='left')
    feat = pd.merge(feat, feat_3, on='userid', how='left')

    print("44444444444")
    print(feat.shape)
    feat.drop('loan_time', axis=1, inplace=True)
    feat = feat.drop_duplicates(subset=['userid'], keep='first')

    print("555555555")
    print(feat.shape)

    target = get_index()
    feat = pd.merge(target, feat, on='userid', how='left')

    print("666666666")
    print(feat.shape)

    feat.to_csv("../cache/feature_bill", index=False)
