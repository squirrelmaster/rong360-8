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
    b_t_k = bill_detail[bill_detail['tm_encode_3'] > 0 & (bill_detail['tm_encode_3'] <= bill_detail['loan_time'])]
    b_t_k_uniq = bill_detail[
        bill_detail['tm_encode_3'] > 0 & (bill_detail['tm_encode_3'] <= bill_detail['loan_time'])].groupby(
        ['userid', 'tm_encode_3', 'bank_id'],
        as_index=False).max()
    a_t_k = bill_detail[bill_detail['tm_encode_3'] > 0 & (bill_detail['tm_encode_3'] < bill_detail['loan_time'])]
    a_t_k_uniq = bill_detail[
        bill_detail['tm_encode_3'] > 0 & (bill_detail['tm_encode_3'] < bill_detail['loan_time'])].groupby(
        ['userid', 'tm_encode_3', 'bank_id'],
        as_index=False).max()
    t_u_k = bill_detail[bill_detail['tm_encode_3'] == 0]
    t_u_k_uniq = bill_detail[bill_detail['tm_encode_3'] == 0].groupby(['userid', 'tm_encode_3', 'bank_id'],
                                                                      as_index=False).max()
    bill_detail_uniq = bill_detail.groupby(['userid', 'tm_encode_3', 'bank_id'], as_index=False).max()

    feature = loan_time
    feature = bill_util(b_t_k, feature, name='b_t_k')
    feature = bill_util(b_t_k_uniq, feature, name='b_t_k_uniq')
    feature = bill_util(a_t_k, feature, name='a_t_k')
    feature = bill_util(a_t_k_uniq, feature, name='a_t_k_uniq')
    feature = bill_util(t_u_k, feature, name='t_u_k')
    feature = bill_util(t_u_k_uniq, feature, name='t_u_k_uniq')
    feature = bill_util(bill_detail_uniq, feature, name='bill_detail_uniq')
    return feature


def bill_util(data, feature, name):
    columns_list = ['prior_account', 'prior_repay', 'credit_limit', 'account_balance',
                    'minimun_repay', 'consume_count', 'account', 'adjust_account', 'circulated_interest',
                    'avaliable_balance', 'cash_limit', 'repay_state']
    for math_methor in ['sum', 'max', 'count', 'min', 'std']:
        temp = data.groupby('userid', as_index=False)[columns_list].agg(math_methor)
        temp.columns = ['userid'] + [name + x + math_methor for x in temp.columns[1:]]
        feature = pd.merge(feature, temp, how='left')
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
    feat = extract_bill_1(bill_detail, loan_time)

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

    feat.to_csv("../cache/feature_bill_2", index=False)
