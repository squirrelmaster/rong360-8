import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import numpy as np


def extract_base_feat():
    user_info_train = pd.read_csv('../data/train/user_info_train_80.csv', header=None)
    user_info_test_A = pd.read_csv('../data/test/user_info_test_A.csv', header=None)
    user_info_test_B = pd.read_csv('../data/test/user_info_test_B.csv', header=None)
    col_names = ['userid', 'sex', 'occupation', 'education', 'marriage', 'household']
    user_info_train.columns = col_names
    user_info_test_A.columns = col_names
    user_info_test_B.columns = col_names
    user_info = pd.concat([user_info_train, user_info_test_A, user_info_test_B])
    user_info.index = user_info['userid']
    user_info.drop('userid', axis=1, inplace=True)

    bank_detail_train = pd.read_csv('../data/train/bank_detail_train_80.csv', header=None)
    bank_detail_test_A = pd.read_csv('../data/test/bank_detail_test_A.csv', header=None)
    bank_detail_test_B = pd.read_csv('../data/test/bank_detail_test_B.csv', header=None)
    col_names = ['userid', 'tm_encode', 'trade_type', 'trade_amount', 'salary_tag']
    bank_detail_train.columns = col_names
    bank_detail_test_A.columns = col_names
    bank_detail_test_B.columns = col_names
    bank_detail = pd.concat([bank_detail_train, bank_detail_test_A, bank_detail_test_B])
    bank_detail['tm_encode'] = bank_detail['tm_encode'] // 86400
    bank_detail_n = (bank_detail.loc[:, ['userid', 'trade_type', 'trade_amount', 'tm_encode']]).groupby(
        ['userid', 'trade_type']).mean()
    bank_detail_n = bank_detail_n.unstack()
    bank_detail_n.columns = ['income', 'outcome', 'income_tm', 'outcome_tm']

    browse_history_train = pd.read_csv('../data/train/browse_history_train_80.csv', header=None)
    browse_history_test_A = pd.read_csv('../data/test/browse_history_test_A.csv', header=None)
    browse_history_test_B = pd.read_csv('../data/test/browse_history_test_B.csv', header=None)
    col_names = ['userid', 'tm_encode_2', 'browse_data', 'browse_tag']
    browse_history_train.columns = col_names
    browse_history_test_A.columns = col_names
    browse_history_test_B.columns = col_names
    browse_history = pd.concat([browse_history_train, browse_history_test_A, browse_history_test_B])
    browse_history_count = browse_history.loc[:, ['userid', 'browse_data']].groupby(['userid']).count()

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
    bill_detail_mean = bill_detail.groupby(['userid']).mean()
    bill_detail_mean.drop('bank_id', axis=1, inplace=True)

    loan_time_train = pd.read_csv('../data/train/loan_time_train_80.csv', header=None)
    loan_time_test_A = pd.read_csv('../data/test/loan_time_test_A.csv', header=None)
    loan_time_test_B = pd.read_csv('../data/test/loan_time_test_B.csv', header=None)
    loan_time = pd.concat([loan_time_train, loan_time_test_A, loan_time_test_B])
    loan_time.columns = ['userid', 'loan_time']
    loan_time['loan_time'] = loan_time['loan_time'] // 86400
    loan_time.index = loan_time['userid']
    loan_time.drop('userid', axis=1, inplace=True)

    target_train = pd.read_csv('../data/train/overdue_train_80.csv', header=None)
    target_test_A = pd.read_csv('../data/test/overdue_test_A.csv', header=None)
    target_test_B = pd.read_csv('../data/test/overdue_test_B.csv', header=None)
    target = pd.concat([target_train, target_test_A, target_test_B])
    target.columns = ['userid', 'label']
    target.index = target['userid']
    target.drop(['userid', 'label'], axis=1, inplace=True)

    feature = pd.merge(target, user_info, how='left', left_index=True, right_index=True)
    feature = pd.merge(feature, bank_detail_n, how='left', left_index=True, right_index=True)
    feature = pd.merge(feature, bill_detail_mean, how='left', left_index=True, right_index=True)
    feature = pd.merge(feature, browse_history_count, how='left', left_index=True, right_index=True)
    feature = pd.merge(feature, loan_time, how='left', left_index=True, right_index=True)

    feature['time'] = feature['loan_time'] - feature['tm_encode_3']
    feature['time1'] = (feature['loan_time'] > feature['tm_encode_3']).astype('int')
    print(feature.shape)

    feature.to_csv("../cache/feature_basic")


if __name__ == '__main__':
    extract_base_feat()
