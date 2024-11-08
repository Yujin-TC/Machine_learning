import pandas as pd
import numpy as np
import Perceptron_algorithm

train_data = pd.read_csv('bank-note/train.csv', header=None)
# process data
raw_train = train_data.values
num_col_train = raw_train.shape[1]
num_row_train = raw_train.shape[0]
train_x = np.copy(raw_train)
train_x[:,num_col_train - 1] = 1
train_y = raw_train[:, num_col_train - 1]
train_y = 2 * train_y - 1

test_data = pd.read_csv('bank-note/test.csv', header=None)
raw_test = test_data.values
num_col_test = raw_test.shape[1]
num_row_test = raw_test.shape[0]
test_x = np.copy(raw_test)
test_x[:,num_col_test - 1] = 1
test_y = raw_test[:, num_col_test - 1]
test_y = 2 * test_y - 1

p = Perceptron_algorithm.Perceptron_algorithm()
#standard algorithm
w = p.standard_predict(train_x, train_y)
w = np.reshape(w, (-1,1))
predict_standard = np.matmul(test_x, w)
predict_standard[predict_standard > 0] = 1
predict_standard[predict_standard <= 0] = -1
error_standard = np.sum(np.abs(predict_standard - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
print('Standard Perceptron: ', error_standard)
print(w)

# voting
#voting every step
#voting_test
c_list, w_list =p.voted_predict_test(train_x, train_y, test_x, test_y)
c_list, w_list =p.voted_predict(train_x, train_y)
c_list = np.reshape(c_list, (-1,1))
print(w_list)
w_list = np.transpose(w_list)
prod = np.matmul(test_x, w_list)
prod[prod >0] = 1
prod[prod <=0] = -1
voted = np.matmul(prod, c_list)
voted[voted >0] = 1
voted[voted<=0] = -1
error_voted = np.sum(np.abs(voted - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
print('Voted Perceptron: ', error_voted)
print(c_list)

# average
w = p.averaged_predict_test(train_x, train_y, test_x, test_y)
w = p.averaged_predict(train_x, train_y)
w = np.reshape(w, (-1,1))
predict_average = np.matmul(test_x, w)
predict_average[predict_average > 0] = 1
predict_average[predict_average <= 0] = -1
error_averaged = np.sum(np.abs(predict_average - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]
print('Averaged Perceptron: ', error_averaged)
print(w)
