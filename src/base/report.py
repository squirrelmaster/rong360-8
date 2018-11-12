from sklearn.cross_validation import train_test_split
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
    feature_useful_2 = pd.read_csv('../cache/feature_useful_2', index_col='userid')
    feature_bill = pd.read_csv('../cache/feature_bill', index_col='userid')
    #feature_bill_2 = pd.read_csv('../cache/feature_bill_2', index_col='userid')
    feature_bank = pd.read_csv('../cache/feature_bank', index_col='userid')
    feature_browse = pd.read_csv('../cache/feature_browse', index_col='userid')

    feature_all = pd.merge(feature_useful, feature_basic, how='left', left_index=True, right_index=True)
    feature_all = pd.merge(feature_all, feature_useful_2, how='left', left_index=True, right_index=True)
    feature_all = pd.merge(feature_all, feature_bill, how='left', left_index=True, right_index=True)
    feature_all = pd.merge(feature_all, feature_bank, how='left', left_index=True, right_index=True)
    feature_all = pd.merge(feature_all, feature_browse, how='left', left_index=True, right_index=True)
   # feature_all = pd.merge(feature_all, feature_bill_2, how='left', left_index=True, right_index=True)
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
              'objective': 'binary:logistic',
              'eval_metric': 'logloss',
              'max_depth': 9,
              'lambda': 20,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'min_child_weight': 2,
              'eta': 0.01,
              'seed': 1000,
              'nthread': 12,
              'silent': 1}

    xgb_train = xgb.DMatrix(train_X, label=train_y)
    xgb_model = xgb.train(params, xgb_train, 1200)
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


def report_xgb_2():
    train_X, train_y, test_X_A, test_y_A, test_X_B, test_y_B = get_feature()
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.2, random_state=0)
    param = {'learning_rate': 0.1, 'n_estimators': 1000, 'max_depth': 7,
             'min_child_weight': 5, 'gamma': 0, 'subsample': 0.8, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.05, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 300
    param['eval_metric'] = "auc"
    plst = list(param.items())
    plst += [('eval_metric', 'logloss')]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    evallist = [(dtest, 'eval'), (dtrain, 'train')]
    print(plst)
    xgb_model = xgb.train(plst, dtrain, num_round, evallist, verbose_eval=50)

    xgb_test_A = xgb.DMatrix(test_X_A)
    pred_test_A = xgb_model.predict(xgb_test_A)
    np.savetxt("../result/xgb_A_2.txt", pred_test_A)
    fpr, tpr, thres = roc_curve(test_y_A, pred_test_A, pos_label=1)
    print(abs(fpr - tpr).max())
    xgb_test_B = xgb.DMatrix(test_X_B)
    pred_test_B = xgb_model.predict(xgb_test_B)
    np.savetxt("../result/xgb_B_2.txt", pred_test_B)
    fpr, tpr, thres = roc_curve(test_y_B, pred_test_B, pos_label=1)
    print(abs(fpr - tpr).max())


def report_lgb():
    train_X, train_y, test_X_A, test_y_A, test_X_B, test_y_B = get_feature()
    lgb_model = lgb.LGBMRegressor(n_estimators=11000, boosting_type="gbdt", learning_rate=0.001)
    lgb_model.fit(train_X, train_y)
    pred_test_A = lgb_model.predict(test_X_A)
    pred_test_A = nom0_1(pred_test_A.reshape(-1, 1))

    np.savetxt("../result/lgb_A.txt", pred_test_A)
    fpr, tpr, thres = roc_curve(test_y_A, pred_test_A, pos_label=1)
    print(abs(fpr - tpr).max())
    pred_test_B = lgb_model.predict(test_X_B)
    pred_test_B = nom0_1(pred_test_B.reshape(-1, 1))
    np.savetxt("../result/lgb_B.txt", pred_test_B)
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


def model_merge():
    lgb_test_A = np.loadtxt("../result/lgb_A.txt")
    lgb_test_B = np.loadtxt("../result/lgb_B.txt")
    lgb_test_A_no = np.loadtxt("../result/lgb_A_no_useful2.txt")
    lgb_test_B_no = np.loadtxt("../result/lgb_B_no_useful2.txt")
    xgb_test_A = np.loadtxt("../result/xgb_A.txt")
    xgb_test_B = np.loadtxt("../result/xgb_B_2.txt")

    lr_test_A = np.loadtxt("../result/lr_A.txt")
    lr_test_B = np.loadtxt("../result/lr_B.txt")
    rf_test_A = np.loadtxt("../result/rf_A.txt")
    rf_test_B = np.loadtxt("../result/rf_B.txt")

    test_y_A = np.loadtxt("../result/label_A.txt")
    test_y_B = np.loadtxt("../result/label_B.txt")
    pred_test_A = 0.6 * lgb_test_A + 0.3 * lgb_test_A_no + 0.1 * lr_test_A
    fpr, tpr, thres = roc_curve(test_y_A, pred_test_A, pos_label=1)
    print(abs(fpr - tpr).max())
    pred_test_B = 0.7 * lgb_test_B + 0.3 * xgb_test_B
    fpr, tpr, thres = roc_curve(test_y_B, pred_test_B, pos_label=1)
    print(abs(fpr - tpr).max())


def model_merge_2():
    lgb_test_A = np.loadtxt("../result/lgb_A.txt")
    xgb_test_A = np.loadtxt("../result/xgb_A_2.txt")
    test_y_A = np.loadtxt("../result/label_A.txt")
    test_A = pd.DataFrame(index=None, columns=['id', 'lgb', 'xgb'])
    test_A['id'] = test_y_A
    test_A['lgb'] = lgb_test_A
    test_A['xgb'] = xgb_test_A

    train_X = test_A[['lgb', 'xgb']]
    train_y = test_A[['id']]

    lr_model = LogisticRegression(C=1.0, penalty='l2')
    lr_model.fit(train_X, train_y)

    lgb_test_B = np.loadtxt("../result/lgb_B.txt")
    xgb_test_B = np.loadtxt("../result/xgb_B_2.txt")
    test_y_B = np.loadtxt("../result/label_B.txt")
    test_B = pd.DataFrame(index=None, columns=['id', 'lgb', 'xgb'])
    test_B['id'] = test_y_B
    test_B['lgb'] = lgb_test_B
    test_B['xgb'] = xgb_test_B

    test_X_B = test_B[['lgb', 'xgb']]
    test_y_B = test_B[['id']]
    pred_test_B = lr_model.predict_proba(test_X_B)[:, 1]

    fpr, tpr, thres = roc_curve(test_y_B, pred_test_B, pos_label=1)
    print(abs(fpr - tpr).max())


def nom0_1(data):
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)


if __name__ == '__main__':
    # report_xgb_2()
    report_xgb()
    # model_merge_2()
