import pandas as pd
import numpy as np
import scipy.optimize as opt


class SVM:
    def __init__(self, C=10, lr=0.1, d=0.1, epoch=100, gamma=0.1):
        self.C = C
        self.lr = lr
        self.d = d
        self.epoch = epoch
        self.gamma = gamma

    def train_predict(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros(dim)
        idx = np.arange(num_sample)
        for t in range(self.epoch):
            np.random.shuffle(idx)
            x = x[idx, :]
            y = y[idx]
            for i in range(num_sample):
                tmp = y[i] * np.sum(np.multiply(w, x[i, :]))
                g = np.copy(w)
                g[dim - 1] = 0
                if tmp <= 1:
                    g = g - self.C * num_sample * y[i] * x[i, :]
                #situation 1
                #lr = self.lr / (1 + self.lr / self.d * t)
                #situation 2
                lr = self.lr / (1 + t)
                w = w - lr * g
        return w

    def object(self, alpha, x, y):
        l = 0
        l = l - np.sum(alpha)
        ayx = np.multiply(np.multiply(np.reshape(alpha, (-1, 1)), np.reshape(y, (-1, 1))), x)
        l = l + 0.5 * np.sum(np.matmul(ayx, np.transpose(ayx)))
        return l

    def constrain(self, alpha, y):
        t = np.matmul(np.reshape(alpha, (1, -1)), np.reshape(y, (-1, 1)))
        return t[0]

    def train_dual(self, x, y):
        num_sample = x.shape[0]
        bounds = [(0, self.C)] * num_sample
        cons = ({'type': 'eq', 'fun': lambda alpha: self.constrain(alpha, y)})
        alpha0 = np.zeros(num_sample)
        res = opt.minimize(lambda alpha: self.object(alpha, x, y), alpha0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'disp': False})

        w = np.sum(np.multiply(np.multiply(np.reshape(res.x, (-1, 1)), np.reshape(y, (-1, 1))), x), axis=0)
        idx = np.where((res.x > 0) & (res.x < self.C))
        b = np.mean(y[idx] - np.matmul(x[idx, :], np.reshape(w, (-1, 1))))
        w = w.tolist()
        w.append(b)
        w = np.array(w)
        return w

    def gaussian_kernel(self, x1, x2, gamma):
        m1 = np.tile(x1, (1, x2.shape[0]))
        m1 = np.reshape(m1, (-1, x1.shape[1]))
        m2 = np.tile(x2, (x1.shape[0], 1))
        k = np.exp(np.sum(np.square(m1 - m2), axis=1) / -gamma)
        k = np.reshape(k, (x1.shape[0], x2.shape[0]))
        return k

    def object_gaussian_kernel(self, alpha, k, y):
        l = 0
        l = l - np.sum(alpha)
        ay = np.multiply(np.reshape(alpha, (-1, 1)), np.reshape(y, (-1, 1)))
        ayay = np.matmul(ay, np.transpose(ay))
        l = l + 0.5 * np.sum(np.multiply(ayay, k))
        return l

    def train_gaussian_kernel(self, x, y):
        num_sample = x.shape[0]
        bounds = [(0, self.C)] * num_sample
        cons = ({'type': 'eq', 'fun': lambda alpha: self.constrain(alpha, y)})
        alpha0 = np.zeros(num_sample)
        k = self.gaussian_kernel(x, x, self.gamma)
        result = opt.minimize(lambda alpha: self.object_gaussian_kernel(alpha, k, y), alpha0, method='SLSQP', bounds=bounds,
                              constraints=cons, options={'disp': False})
        return result.x

    def predict_gaussian_kernel(self, alpha, x0, y0, x):
        k = self.gaussian_kernel(x0, x, self.gamma)
        k = np.multiply(np.reshape(y0, (-1, 1)), k)
        y = np.sum(np.multiply(np.reshape(alpha, (-1, 1)), k), axis=0)
        y = np.reshape(y, (-1, 1))
        y[y > 0] = 1
        y[y <= 0] = -1
        return y
