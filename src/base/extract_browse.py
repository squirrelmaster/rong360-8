import pandas as pd
import numpy as np


def get_feat():
    browse_history_train = pd.read_csv('../data/train/browse_history_train_80.csv', header=None)
    browse_history_test_A = pd.read_csv('../data/test/browse_history_test_A.csv', header=None)
    browse_history_test_B = pd.read_csv('../data/test/browse_history_test_B.csv', header=None)
    col_names = ['userid', 'tm_encode_2', 'browse_data', 'browse_tag']
    browse_history_train.columns = col_names
    browse_history_test_A.columns = col_names
    browse_history_test_B.columns = col_names
    browse_history = pd.concat([browse_history_train, browse_history_test_A, browse_history_test_B])
    browse_history['tm_encode_2'] = browse_history['tm_encode_2'] // 86400
    browse_history['browse_tag'] = browse_history['browse_tag']
    loan_time_train = pd.read_csv('../data/train/loan_time_train_80.csv', header=None)
    loan_time_test_A = pd.read_csv('../data/test/loan_time_test_A.csv', header=None)
    loan_time_test_B = pd.read_csv('../data/test/loan_time_test_B.csv', header=None)
    loan_time = pd.concat([loan_time_train, loan_time_test_A, loan_time_test_B])
    loan_time.columns = ['userid', 'loan_time']
    loan_time['loan_time'] = loan_time['loan_time'] // 86400
    browse_history = pd.merge(browse_history, loan_time, on='userid')
    return browse_history, loan_time


def extract_browse(browse_history, loan_time):
    # ==========================放款前==========================
    feature = loan_time
    d = browse_history

    d = d[d['tm_encode_2'] <= d['loan_time']]

    gb = d.groupby('userid', as_index=False)
    x1 = gb['browse_data'].agg(
        {'b_browse_data_sum': np.sum, 'b_browse_data_mean': np.mean, 'b_browse_data_max': np.max,
         'b_browse_data_min': np.min, 'b_browse_data_std': np.std,'b_browse_data_var': np.var,
         'b_browse_data_median': np.median})
    x2 = gb['browse_tag'].agg({'b_browse_tag_sum': np.sum, 'b_browse_tag_mean': np.mean, 'b_browse_tag_max': np.max,
                               'b_browse_tag_min': np.min, 'b_browse_tag_std': np.std,
                               'b_browse_tag_median': np.median})
    xx = gb['browse_tag'].apply(lambda x: np.unique(x).size)
    x3 = gb['browse_tag'].agg({'b_browse_tag_size': np.size})
    x3['b_browse_tag_uniq'] = xx

    feature = pd.merge(feature, x1, on='userid', how='left')
    feature = pd.merge(feature, x2, on='userid', how='left')
    feature = pd.merge(feature, x3, on='userid', how='left')

    d = d.dropna(subset=['browse_tag'], axis=0)
    d['browse_tag'] = d['browse_tag'].astype(int)
    temp = pd.get_dummies(d, columns=['browse_tag'])

    gb = temp.groupby('userid', as_index=False)
    x1 = gb['browse_tag_1'].agg({'b_browse_tag_1sum': np.sum})
    x2 = gb['browse_tag_2'].agg({'b_browse_tag_2sum': np.sum})
    x3 = gb['browse_tag_3'].agg({'b_browse_tag_3sum': np.sum})
    x4 = gb['browse_tag_4'].agg({'b_browse_tag_4sum': np.sum})
    x5 = gb['browse_tag_5'].agg({'b_browse_tag_5sum': np.sum})
    x6 = gb['browse_tag_6'].agg({'b_browse_tag_6sum': np.sum})
    x7 = gb['browse_tag_7'].agg({'b_browse_tag_7sum': np.sum})
    x8 = gb['browse_tag_8'].agg({'b_browse_tag_8sum': np.sum})
    x9 = gb['browse_tag_9'].agg({'b_browse_tag_9sum': np.sum})
    x10 = gb['browse_tag_10'].agg({'b_browse_tag_10sum': np.sum})
    x11 = gb['browse_tag_11'].agg({'b_browse_tag_11sum': np.sum})

    feature = pd.merge(feature, x1, on='userid', how='left')
    feature = pd.merge(feature, x2, on='userid', how='left')
    feature = pd.merge(feature, x3, on='userid', how='left')
    feature = pd.merge(feature, x4, on='userid', how='left')
    feature = pd.merge(feature, x5, on='userid', how='left')
    feature = pd.merge(feature, x6, on='userid', how='left')
    feature = pd.merge(feature, x7, on='userid', how='left')
    feature = pd.merge(feature, x8, on='userid', how='left')
    feature = pd.merge(feature, x9, on='userid', how='left')
    feature = pd.merge(feature, x10, on='userid', how='left')
    feature = pd.merge(feature, x11, on='userid', how='left')

    feature1 = feature.drop('loan_time', axis=1)
    print("11111111111111")
    print(feature1.shape)
    # ==========================放款后==========================
    feature = loan_time
    d = browse_history

    d = d[d['tm_encode_2'] > d['loan_time']]

    gb = d.groupby('userid', as_index=False)
    x1 = gb['browse_data'].agg(
        {'a_browse_data_sum': np.sum, 'a_browse_data_mean': np.mean, 'a_browse_data_max': np.max,
         'a_browse_data_min': np.min, 'a_browse_data_std': np.std,'a_browse_data_var': np.var,
         'a_browse_data_median': np.median})
    x2 = gb['browse_tag'].agg({'a_browse_tag_sum': np.sum, 'a_browse_tag_mean': np.mean, 'a_browse_tag_max': np.max,
                               'a_browse_tag_min': np.min, 'a_browse_tag_std': np.std,
                               'a_browse_tag_median': np.median})
    xx = gb['browse_tag'].apply(lambda x: np.unique(x).size)
    x3 = gb['browse_tag'].agg({'a_browse_tag_size': np.size})
    x3['a_browse_tag_uniq'] = xx

    feature = pd.merge(feature, x1, on='userid', how='left')
    feature = pd.merge(feature, x2, on='userid', how='left')
    feature = pd.merge(feature, x3, on='userid', how='left')

    d = d.dropna(subset=['browse_tag'], axis=0)
    d['browse_tag'] = d['browse_tag'].astype(int)
    temp = pd.get_dummies(d, columns=['browse_tag'])
    gb = temp.groupby('userid', as_index=False)
    x1 = gb['browse_tag_1'].agg({'a_browse_tag_1sum': np.sum})
    x2 = gb['browse_tag_2'].agg({'a_browse_tag_2sum': np.sum})
    x3 = gb['browse_tag_3'].agg({'a_browse_tag_3sum': np.sum})
    x4 = gb['browse_tag_4'].agg({'a_browse_tag_4sum': np.sum})
    x5 = gb['browse_tag_5'].agg({'a_browse_tag_5sum': np.sum})
    x6 = gb['browse_tag_6'].agg({'a_browse_tag_6sum': np.sum})
    x7 = gb['browse_tag_7'].agg({'a_browse_tag_7sum': np.sum})
    x8 = gb['browse_tag_8'].agg({'a_browse_tag_8sum': np.sum})
    x9 = gb['browse_tag_9'].agg({'a_browse_tag_9sum': np.sum})
    x10 = gb['browse_tag_10'].agg({'a_browse_tag_10sum': np.sum})
    x11 = gb['browse_tag_11'].agg({'a_browse_tag_11sum': np.sum})

    feature = pd.merge(feature, x1, on='userid', how='left')
    feature = pd.merge(feature, x2, on='userid', how='left')
    feature = pd.merge(feature, x3, on='userid', how='left')
    feature = pd.merge(feature, x4, on='userid', how='left')
    feature = pd.merge(feature, x5, on='userid', how='left')
    feature = pd.merge(feature, x6, on='userid', how='left')
    feature = pd.merge(feature, x7, on='userid', how='left')
    feature = pd.merge(feature, x8, on='userid', how='left')
    feature = pd.merge(feature, x9, on='userid', how='left')
    feature = pd.merge(feature, x10, on='userid', how='left')
    feature = pd.merge(feature, x11, on='userid', how='left')
    feature2 = feature.drop('loan_time', axis=1)

    print(feature2.shape)

    feature = pd.merge(feature1, feature2, on='userid', how='left')

    print("22222222222222")
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
    browse_history, loan_time = get_feat()
    feat = extract_browse(browse_history, loan_time)

    print("2222222222")
    print(feat.shape)
    feat = feat.drop_duplicates(subset=['userid'], keep='first')

    print("333333333333333")
    print(feat.shape)

    target = get_index()
    feat = pd.merge(target, feat, on='userid', how='left')

    print("44444444444444444")
    print(feat.shape)

    feat.to_csv("../cache/feature_browse", index=False)
