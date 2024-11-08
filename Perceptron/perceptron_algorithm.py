from ftplib import error_temp

import pandas as pd
import numpy as np


class Perceptron_algorithm:
    def __init__(self, learning_rate=0.2, Epoch=10):
        self.learning_rate = learning_rate
        self.Epoch = Epoch

    def standard_predict(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros(dim)
        idx = np.arange(num_sample)
        for t in range(self.Epoch):
            np.random.shuffle(idx)
            x = x[idx, :]
            y = y[idx]
            for i in range(num_sample):
                tmp = np.sum(x[i] * w)
                if not (tmp * y[i] > 0):
                    w = w + self.learning_rate * y[i] * x[i]
        return w

    def voted_predict(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros(dim)
        idx = np.arange(num_sample)
        c_list = np.array([])
        w_list = np.array([])
        c = 0
        for t in range(self.Epoch):
            np.random.shuffle(idx)
            x = x[idx, :]
            y = y[idx]
            for i in range(num_sample):
                tmp = np.sum(x[i] * w)
                if not (tmp * y[i] > 0):
                    w_list = np.append(w_list, w)
                    c_list = np.append(c_list, c)
                    w = w + self.learning_rate * y[i] * x[i]
                    c = 1
                else:
                    c = c + 1
        num = c_list.shape[0]
        w_list = np.reshape(w_list, (num, -1))
        return c_list, w_list,

    def averaged_predict(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros(dim)
        a = np.zeros(dim)
        idx = np.arange(num_sample)
        for t in range(self.Epoch):
            np.random.shuffle(idx)
            x = x[idx, :]
            y = y[idx]
            for i in range(num_sample):
                tmp = np.sum(x[i] * w)
                if not (tmp * y[i] > 0):
                    w = w + self.learning_rate * y[i] * x[i]
                a = a + w
        return a

    def averaged_predict_test(self, x_train, y_train, x_test, y_test):
        num_sample = x_train.shape[0]
        dim = x_train.shape[1]
        w = np.zeros(dim)
        a = np.zeros(dim)
        idx = np.arange(num_sample)
        for t in range(self.Epoch):
            np.random.shuffle(idx)
            x = x_train[idx, :]
            y = y_train[idx]
            for i in range(num_sample):
                tmp = np.sum(x[i] * w)
                if not (tmp * y[i] > 0):
                    w = w + self.learning_rate * y[i] * x[i]
                a = a + w
            self.calc_error_averaged(a, x_test, y_test)
        return a

    def calc_error_averaged(self, w, test_x, test_y):
        w = np.reshape(w, (-1, 1))
        predict_average = np.matmul(test_x, w)
        predict_average[predict_average > 0] = 1
        predict_average[predict_average <= 0] = -1
        error_averaged = np.sum(np.abs(predict_average - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
        print("Averaged Error: ", error_averaged)

    def voted_predict_test(self, x, y, x_test, y_test):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros(dim)
        idx = np.arange(num_sample)
        c_list = np.array([])
        w_list = np.array([])
        c = 0
        for t in range(self.Epoch):
            np.random.shuffle(idx)
            x = x[idx, :]
            y = y[idx]
            for i in range(num_sample):
                tmp = np.sum(x[i] * w)
                if not (tmp * y[i] > 0):
                    w_list = np.append(w_list, w)
                    c_list = np.append(c_list, c)
                    w = w + self.learning_rate * y[i] * x[i]
                    c = 1
                else:
                    c = c + 1
            w_list_tmp = w_list.copy()
            num = c_list.shape[0]
            w_list_tmp = np.reshape(w_list_tmp, (num, -1))
            self.calc_error_voted(y_test, x_test, w_list_tmp, c_list)
        num = c_list.shape[0]
        w_list = np.reshape(w_list, (num, -1))
        return c_list, w_list

    def calc_error_voted(self, test_y, test_x, w_list, c_list):
        c_list = np.reshape(c_list, (-1, 1))
        #print(w_list)
        w_list = np.transpose(w_list)
        prod = np.matmul(test_x, w_list)
        prod[prod > 0] = 1
        prod[prod <= 0] = -1
        voted = np.matmul(prod, c_list)
        voted[voted > 0] = 1
        voted[voted <= 0] = -1
        error_voted = np.sum(np.abs(voted - np.reshape(test_y, (-1, 1)))) / 2 / test_y.shape[0]
        print("Error: ", error_voted)

