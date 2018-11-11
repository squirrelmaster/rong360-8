from minepy import MINE
import matplotlib.pyplot as plt
import numpy as np


def mic():
    lgb_test_A = np.loadtxt("../result/lgb_A.txt")
    lgb_test_A_no = np.loadtxt("../result/lgb_A_no_useful2.txt")
    xgb_test_A = np.loadtxt("../result/xgb_A.txt")

    lr_test_A = np.loadtxt("../result/lr_A.txt")

    rf_test_A = np.loadtxt("../result/rf_A.txt")

    res = [lgb_test_A, lgb_test_A_no, xgb_test_A, lr_test_A, rf_test_A]

    cm = []
    for i in range(5):
        tmp = []
        for j in range(5):
            m = MINE()
            m.compute_score(res[i], res[j])
            tmp.append(m.mic())
        cm.append(tmp)
    return cm


def plot_confusion_matrix(cm, title='mic', cmap=plt.cm.Blues):
    fs = ['lgb','lgb_no', 'xgb', 'lr', 'rf']
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, fs, rotation=45)
    plt.yticks(tick_marks, fs)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_confusion_matrix(mic())
