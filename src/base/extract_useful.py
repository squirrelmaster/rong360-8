import pandas as pd
from sklearn.metrics import roc_curve
import lightgbm as lgb


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
    t3 = train_test[(train_test['tm_encode_3'] > train_test['loan_time'] + 20)].groupby("userid", as_index=False)
    t4 = train_test[(train_test['tm_encode_3'] > train_test['loan_time'] + 45)].groupby("userid", as_index=False)

    t1 = t1['tm_encode_3'].agg({'t1': 'count'})
    t2 = t2['tm_encode_3'].agg({'t2': 'count'})
    t3 = t3['tm_encode_3'].agg({'t3': 'count'})
    t4 = t4['tm_encode_3'].agg({'t4': 'count'})

    target_train = pd.read_csv('../data/train/overdue_train_80.csv', header=None)
    target_test_A = pd.read_csv('../data/test/overdue_test_A.csv', header=None)
    target_test_B = pd.read_csv('../data/test/overdue_test_B.csv', header=None)
    target_train.columns = ['userid', 'label']
    target_test_A.columns = ['userid', 'label']
    target_test_B.columns = ['userid', 'label']
    target = pd.concat([target_train, target_test_A, target_test_B])

    train_test = pd.merge(target, t1, how='left', on="userid")
    train_test = pd.merge(train_test, t2, how='left', on="userid")
    train_test = pd.merge(train_test, t3, how='left', on="userid")
    train_test = pd.merge(train_test, t4, how='left', on="userid")

    train_test = train_test.fillna(0)
    train_test.index = train_test['userid']
    train_test.drop(['userid'], axis=1, inplace=True)

    target.index = target['userid']
    target.drop('userid', axis=1, inplace=True)

    print(train_test.shape)
    print(target.shape)

    train_test.to_csv("../cache/feature_useful")


def report():
    feature = pd.read_csv("../cache/feature_useful", index_col='userid')
    print(feature.head())
    train = feature.iloc[0: 44476, :]
    test_A = feature.iloc[44476:50037, :]
    test_B = feature.iloc[50037:, :]

    train_X = train.drop(['label'], axis=1, )
    test_X_A = test_A.drop(['label'], axis=1, )
    test_X_B = test_B.drop(['label'], axis=1, )
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
#    report()
