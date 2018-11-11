from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_feature():
    feature_basic = pd.read_csv("../cache/feature_basic", index_col='userid')
    feature_useful = pd.read_csv('../cache/feature_useful', index_col='userid')
    # feature_useful_2 = pd.read_csv('../cache/feature_useful_2', index_col='userid')
    feature_bill = pd.read_csv('../cache/feature_bill', index_col='userid')
    feature_bank = pd.read_csv('../cache/feature_bank', index_col='userid')
    feature_browse = pd.read_csv('../cache/feature_browse', index_col='userid')

    feature_all = pd.merge(feature_useful, feature_basic, how='left', left_index=True, right_index=True)
    # feature_all = pd.merge(feature_all, feature_useful_2, how='left', left_index=True, right_index=True)
    feature_all = pd.merge(feature_all, feature_bill, how='left', left_index=True, right_index=True)
    feature_all = pd.merge(feature_all, feature_bank, how='left', left_index=True, right_index=True)
    feature_all = pd.merge(feature_all, feature_browse, how='left', left_index=True, right_index=True)
    feature_all = feature_all.fillna(0)
    feature_all = set_dummies(feature_all)

    print(feature_all.shape)
    train = feature_all.iloc[0: 44476, :]
    test_A = feature_all.iloc[44476:50037, :]
    test_B = feature_all.iloc[50037:, :]

    train_X = train.drop(['label'], axis=1)
    test_X_A = test_A.drop(['label'], axis=1)
    test_X_B = test_B.drop(['label'], axis=1)
    train_y = train['label']
    test_y_A = test_A['label']
    test_y_B = test_B['label']

    np.savetxt("../result/label_A.txt", test_y_A)
    np.savetxt("../result/label_B.txt", test_y_B)
    return train_X, train_y, test_X_A, test_y_A, test_X_B, test_y_B


def report_lr():
    train_X, train_y, test_X_A, test_y_A, test_X_B, test_y_B = get_feature()
    lr_model = LogisticRegression(C=1.0, penalty='l2')
    lr_model.fit(train_X, train_y)
    pred_test_A = lr_model.predict_proba(test_X_A)[:, 1]
    np.savetxt("../result/lr_A.txt", pred_test_A)
    fpr, tpr, thres = roc_curve(test_y_A, pred_test_A, pos_label=1)
    print(abs(fpr - tpr).max())
    pred_test_B = lr_model.predict_proba(test_X_B)[:, 1]
    np.savetxt("../result/lr_B.txt", pred_test_B)
    fpr, tpr, thres = roc_curve(test_y_B, pred_test_B, pos_label=1)
    print(abs(fpr - tpr).max())


def report_rf():
    train_X, train_y, test_X_A, test_y_A, test_X_B, test_y_B = get_feature()
    rf_model = RandomForestRegressor()
    rf_model.fit(train_X, train_y)
    pred_test_A = rf_model.predict(test_X_A)
    np.savetxt("../result/rf_A.txt", pred_test_A)
    fpr, tpr, thres = roc_curve(test_y_A, pred_test_A, pos_label=1)
    print(abs(fpr - tpr).max())
    pred_test_B = rf_model.predict(test_X_B)
    np.savetxt("../result/rf_B.txt", pred_test_B)
    fpr, tpr, thres = roc_curve(test_y_B, pred_test_B, pos_label=1)
    print(abs(fpr - tpr).max())


def report_svr():
    train_X, train_y, test_X_A, test_y_A, test_X_B, test_y_B = get_feature()
    svr_model = SVR()
    svr_model.fit(train_X, train_y)
    pred_test_A = svr_model.predict(test_X_A)
    fpr, tpr, thres = roc_curve(test_y_A, pred_test_A, pos_label=1)
    print(abs(fpr - tpr).max())
    pred_test_B = svr_model.predict(test_X_B)
    fpr, tpr, thres = roc_curve(test_y_B, pred_test_B, pos_label=1)
    print(abs(fpr - tpr).max())


def report_xgb():
    train_X, train_y, test_X_A, test_y_A, test_X_B, test_y_B = get_feature()
    params = {'booster': 'gbtree',
              'objective': 'rank:pairwise',
              'eval_metric': 'auc',
              'max_depth': 7,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'min_child_weight': 2,
              'eta': 0.001,
              'seed': 1000,
              'nthread': 12,
              'silent': 1}

    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_model = xgb.train(params, xgb_train, 6000, verbose_eval=10)
    xgb_test_A = xgb.DMatrix(test_X_A)
    pred_test_A = xgb_model.predict(xgb_test_A)
    np.savetxt("../result/xgb_A.txt", pred_test_A)
    fpr, tpr, thres = roc_curve(test_y_A, pred_test_A, pos_label=1)
    print(abs(fpr - tpr).max())
    xgb_test_B = xgb.DMatrix(test_X_B)
    pred_test_B = xgb_model.predict(xgb_test_B)
    np.savetxt("../result/xgb_B.txt", pred_test_B)
    fpr, tpr, thres = roc_curve(test_y_B, pred_test_B, pos_label=1)
    print(abs(fpr - tpr).max())


def report_lgb():
    train_X, train_y, test_X_A, test_y_A, test_X_B, test_y_B = get_feature()
    lgb_model = lgb.LGBMRegressor(n_estimators=11000, boosting_type="gbdt", learning_rate=0.001)
    lgb_model.fit(train_X, train_y)
    pred_test_A = lgb_model.predict(test_X_A)
    pred_test_A = nom0_1(pred_test_A.reshape(-1, 1))

    np.savetxt("../result/lgb_A_no_useful2.txt", pred_test_A)
    fpr, tpr, thres = roc_curve(test_y_A, pred_test_A, pos_label=1)
    print(abs(fpr - tpr).max())
    pred_test_B = lgb_model.predict(test_X_B)
    pred_test_B = nom0_1(pred_test_B.reshape(-1, 1))
    np.savetxt("../result/lgb_B_no_useful2.txt", pred_test_B)
    fpr, tpr, thres = roc_curve(test_y_B, pred_test_B, pos_label=1)
    print(abs(fpr - tpr).max())


def set_dummies(data):
    category_col = ['sex', 'occupation', 'education', 'marriage', 'household']
    for col in category_col:
        data[col] = data[col].astype('category')
        dummy = pd.get_dummies(data[col], prefix=col)
        data = data.drop(col, axis=1)
        data = data.join(dummy)
    return data


def modle_merge():
    lgb_test_A = np.loadtxt("../result/lgb_A.txt")
    lgb_test_B = np.loadtxt("../result/lgb_B.txt")
    lgb_test_A_no = np.loadtxt("../result/lgb_A_no_useful2.txt")
    lgb_test_B_no = np.loadtxt("../result/lgb_B_no_useful2.txt")
    xgb_test_A = np.loadtxt("../result/xgb_A.txt")
    xgb_test_B = np.loadtxt("../result/xgb_B.txt")

    lr_test_A = np.loadtxt("../result/lr_A.txt")
    lr_test_B = np.loadtxt("../result/lr_B.txt")
    rf_test_A = np.loadtxt("../result/rf_A.txt")
    rf_test_B = np.loadtxt("../result/rf_B.txt")

    test_y_A = np.loadtxt("../result/label_A.txt")
    test_y_B = np.loadtxt("../result/label_B.txt")
    pred_test_A = 0.6 * lgb_test_A + 0.3 * lgb_test_A_no +0.1*lr_test_A
    fpr, tpr, thres = roc_curve(test_y_A, pred_test_A, pos_label=1)
    print(abs(fpr - tpr).max())
    pred_test_B = 0.6 * lgb_test_B + 0.1 * xgb_test_B + 0.2 * lr_test_B + 0.1 * rf_test_B
    fpr, tpr, thres = roc_curve(test_y_B, pred_test_B, pos_label=1)
    print(abs(fpr - tpr).max())


def nom0_1(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


if __name__ == '__main__':
    # report_lgb()
    modle_merge()
