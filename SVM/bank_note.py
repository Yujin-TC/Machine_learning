# %%
import pandas as pd
import numpy as np
import SVM

train_data = pd.read_csv('bank_note/train.csv', header=None)
raw = train_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
train_x = np.copy(raw)
train_x[:, num_col - 1] = 1
train_y = raw[:, num_col - 1]
train_y = 2 * train_y - 1

test_data = pd.read_csv('bank_note/test.csv', header=None)
raw = test_data.values
num_col = raw.shape[1]
num_row = raw.shape[0]
test_x = np.copy(raw)
test_x[:, num_col - 1] = 1
test_y = raw[:, num_col - 1]
test_y = 2 * test_y - 1

C_set = np.array([100, 500, 700])
#C_set = np.array([500])
C_set = C_set / 873
gamma_set = np.array([0.01, 0.1, 0.5, 1, 5, 100])
svm = SVM.SVM()
for C in C_set:
    svm.C
    w = svm.train_predict(train_x, train_y)
    w = np.reshape(w, (5, 1))

    pred = np.matmul(train_x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    train_error = np.sum(np.abs(pred - np.reshape(train_y, (-1, 1)))) / 2 / train_y.shape[0]

    pred = np.matmul(test_x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    test_error = np.sum(np.abs(pred - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
    print('train_error: ', train_error, ' test_error: ', test_error)
    w = np.reshape(w, (1, -1))

# dual form
    w = svm.train_dual(train_x[:, [x for x in range(num_col - 1)]], train_y)
    w = np.reshape(w, (5, 1))

    pred = np.matmul(train_x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    train_error = np.sum(np.abs(pred - np.reshape(train_y, (-1, 1)))) / 2 / train_y.shape[0]

    pred = np.matmul(test_x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1

    test_error = np.sum(np.abs(pred - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
    print('Dual SVM train_error: ', train_error, ' test_error: ', test_error)

# gaussian kernel
    c = 0
    for gamma in gamma_set:
        print('gamma is: ', gamma)
        svm.gamma
        alpha = svm.train_gaussian_kernel(train_x[:, [x for x in range(num_col - 1)]], train_y)
        idx = np.where(alpha > 0)[0]
        # train
        y = svm.predict_gaussian_kernel(alpha, train_x[:, [x for x in range(num_col - 1)]], train_y,
                                        train_x[:, [x for x in range(num_col - 1)]])
        train_error = np.sum(np.abs(y - np.reshape(train_y, (-1, 1)))) / 2 / train_y.shape[0]

        # test
        y = svm.predict_gaussian_kernel(alpha, train_x[:, [x for x in range(num_col - 1)]], train_y,
                                        test_x[:, [x for x in range(num_col - 1)]])
        test_error = np.sum(np.abs(y - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
        print('nonlinear SVM train_error: ', train_error, ' test_error: ', test_error)

        if c > 0:
            repeat = len(np.intersect1d(idx, old_idx))
            print('repeat is: ', repeat)
        c = c + 1
        old_idx = idx
