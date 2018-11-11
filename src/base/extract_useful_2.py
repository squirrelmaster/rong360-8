import pandas as pd
from sklearn.metrics import roc_curve
import lightgbm as lgb
import numpy as np


def extract_feat():
    bill_detail_train = pd.read_csv('../data/train/bill_detail_train_80.csv', header=None)
    bill_detail_test_A = pd.read_csv('../data/test/bill_detail_test_A.csv', header=None)
    bill_detail_test_B = pd.read_csv('../data/test/bill_detail_test_B.csv', header=None)
    col_names = ['userid', 'tm_encode_3', 'bank_id', 'prior_account', 'prior_repay', 'credit_limit', 'account_balance',
                 'minimun_repay', 'consume_count', 'account', 'adjust_account', 'circulated_interest',
                 'avaliable_balance', 'cash_limit', 'repay_state']
    bill_detail_train.columns = col_names
    bill_detail_test_A.columns = col_names
    bill_detail_test_B.columns = col_names
    bill_detail_train['tm_encode_3'] = bill_detail_train['tm_encode_3'] // 86400
    bill_detail_test_A['tm_encode_3'] = bill_detail_test_A['tm_encode_3'] // 86400
    bill_detail_test_B['tm_encode_3'] = bill_detail_test_B['tm_encode_3'] // 86400
    bill_detail_train.drop(
        ['bank_id', 'prior_account', 'prior_repay', 'credit_limit', 'account_balance', 'minimun_repay', 'consume_count',
         'account', 'adjust_account', 'circulated_interest', 'avaliable_balance', 'cash_limit', 'repay_state'], axis=1,
        inplace=True)
    bill_detail_test_A.drop(
        ['bank_id', 'prior_account', 'prior_repay', 'credit_limit', 'account_balance', 'minimun_repay', 'consume_count',
         'account', 'adjust_account', 'circulated_interest', 'avaliable_balance', 'cash_limit', 'repay_state'], axis=1,
        inplace=True)
    bill_detail_test_B.drop(
        ['bank_id', 'prior_account', 'prior_repay', 'credit_limit', 'account_balance', 'minimun_repay', 'consume_count',
         'account', 'adjust_account', 'circulated_interest', 'avaliable_balance', 'cash_limit', 'repay_state'], axis=1,
        inplace=True)

    bill_detail = pd.concat([bill_detail_train, bill_detail_test_A, bill_detail_test_B])

    loan_time_train = pd.read_csv('../data/train/loan_time_train_80.csv', header=None)
    loan_time_test_A = pd.read_csv('../data/test/loan_time_test_A.csv', header=None)
    loan_time_test_B = pd.read_csv('../data/test/loan_time_test_B.csv', header=None)
    loan_time_train.columns = ['userid', 'loan_time']
    loan_time_test_A.columns = ['userid', 'loan_time']
    loan_time_test_B.columns = ['userid', 'loan_time']
    loan_time_train['loan_time'] = loan_time_train['loan_time'] // 86400
    loan_time_test_A['loan_time'] = loan_time_test_A['loan_time'] // 86400
    loan_time_test_B['loan_time'] = loan_time_test_B['loan_time'] // 86400

    loan_time = pd.concat([loan_time_train, loan_time_test_A, loan_time_test_B])

    train_test = pd.merge(bill_detail, loan_time, how='inner', on='userid')
    print(train_test.head())

    t1 = train_test[(train_test['tm_encode_3'] > train_test['loan_time'])].groupby("userid", as_index=False)
    t2 = train_test[(train_test['tm_encode_3'] > train_test['loan_time'] + 1)].groupby("userid", as_index=False)
    t3 = train_test[(train_test['tm_encode_3'] > train_test['loan_time'] + 2)].groupby("userid", as_index=False)
    # t4 = train_test[(train_test['tm_encode_3'] > train_test['loan_time'] + 3)].groupby("userid", as_index=False)

    x = t1['tm_encode_3'].apply(lambda x: np.unique(x).size)
    x1 = t1['tm_encode_3'].agg({'u_f_1': 'count'})
    x1['x1'] = x

    x = t2['tm_encode_3'].apply(lambda x: np.unique(x).size)
    x2 = t2['tm_encode_3'].agg({'u_f_2': 'count'})
    x2['x2'] = x

    x = t3['tm_encode_3'].apply(lambda x: np.unique(x).size)
    x3 = t3['tm_encode_3'].agg({'u_f_3': 'count'})
    x3['x3'] = x

    target_train = pd.read_csv('../data/train/overdue_train_80.csv', header=None)
    target_test_A = pd.read_csv('../data/test/overdue_test_A.csv', header=None)
    target_test_B = pd.read_csv('../data/test/overdue_test_B.csv', header=None)
    target = pd.concat([target_train, target_test_A, target_test_B])
    target.columns = ['userid', 'label']
    target = target.drop('label', axis=1)

    feature = pd.merge(target, x1, how='left', on="userid")
    feature = pd.merge(feature, x2, how='left', on="userid")
    feature = pd.merge(feature, x3, how='left', on="userid")

    feature['u_f'] = (feature['x1'] + 1) * (feature['x2'] + 1) * (feature['x3'] + 1) / 3

    feature = feature.fillna(0)

    feature.to_csv("../cache/feature_useful_2", index=False)
    print(feature.shape)

    print(feature.head())


def report():
    feature = pd.read_csv("../cache/feature_useful_2", index_col='userid')
    feature1 = pd.read_csv("../cache/feature_useful", index_col='userid')
    feature = pd.merge(feature, feature1, left_index=True, right_index=True)
    print(feature.head())
    train = feature.iloc[0: 44476, :]
    test_A = feature.iloc[44476:50037, :]
    test_B = feature.iloc[50037:, :]

    train_X = train.drop(['label'], axis=1)
    test_X_A = test_A.drop(['label'], axis=1)
    test_X_B = test_B.drop(['label'], axis=1)
    train_y = train['label']
    test_y_A = test_A['label']
    test_y_B = test_B['label']
    lgb_model = lgb.LGBMRegressor(n_estimators=6000, boosting_type="gbdt", learning_rate=0.001)
    lgb_model.fit(train_X, train_y)
    pred_test_A = lgb_model.predict(test_X_A)
    pred_test_B = lgb_model.predict(test_X_B)
    fpr, tpr, thres = roc_curve(test_y_A, pred_test_A, pos_label=1)
    print(abs(fpr - tpr).max())
    fpr, tpr, thres = roc_curve(test_y_B, pred_test_B, pos_label=1)
    print(abs(fpr - tpr).max())


if __name__ == '__main__':
    extract_feat()
    report()
